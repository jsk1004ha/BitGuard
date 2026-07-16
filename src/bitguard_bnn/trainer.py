from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import random
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from .cascade import (
    apply_boolean_fast_path,
    cascade_operation_summary,
    route_with_temporal_state,
    tune_boolean_fast_path,
    tune_exit_threshold,
)
from .config import (
    _validated_selection_weights,
    create_run_dir,
    environment_manifest,
    load_config,
    resolve_path,
    save_json,
    save_yaml,
    seed_everything,
)
from .constants import CANONICAL_LABELS
from .data import (
    DataSplit,
    load_dataset,
    make_cross_split,
    make_split,
    validate_labels,
)
from .metrics import (
    benchmark_torch_model,
    calibrate_fixed_fpr_thresholds,
    classification_metrics,
    confusion_frame,
    estimate_dense_operations,
    make_plots,
)
from .preprocess import FeaturePreprocessor, class_weights
from .state import replay_predictions


@dataclass
class NeuralFitResult:
    model: Any
    history: pd.DataFrame
    best_validation_score: float
    best_epoch: int


_NEURAL_HISTORY_FIELDS = frozenset(
    {
        "epoch",
        "learning_rate",
        "validation_macro_f1",
        "validation_macro_auprc",
        "validation_attack_recall",
        "validation_selection_score",
        "train_loss",
        "train_detection",
        "train_feature_cost",
        "train_fn",
        "train_fp",
    }
)
_NEURAL_VALIDATION_FIELDS = (
    "validation_macro_f1",
    "validation_macro_auprc",
    "validation_attack_recall",
    "validation_selection_score",
)


def _process_resource_summary(artifact: Path) -> dict[str, Any]:
    try:
        import resource

        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss_bytes: int | None = maximum_rss if sys.platform == "darwin" else maximum_rss * 1024
    except ImportError:
        rss_bytes = None
    return {
        "artifact_file_bytes": artifact.stat().st_size,
        "peak_process_rss_bytes_including_data_and_runtime": rss_bytes,
        "energy_per_decision_joules": None,
        "energy_note": "Measure on the target edge device with an external power monitor.",
    }


def _select_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a Torch checkpoint durably and remove every failed temporary."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_torch_load(path: Path, device: Any) -> dict[str, Any]:
    """Load tensor/primitives-only checkpoints from one pinned regular file."""

    import torch

    version = str(getattr(torch, "__version__", ""))
    match = re.fullmatch(
        r"\s*(\d+)\.(\d+)(?:\.\d+)?(?:[A-Za-z0-9_.+-]*)\s*", version
    )
    if match is None:
        raise RuntimeError(
            "unable to verify a safe PyTorch version before checkpoint loading"
        )
    if (int(match.group(1)), int(match.group(2))) < (2, 6):
        raise RuntimeError(
            "checkpoint loading requires PyTorch 2.6 or newer for CVE-2025-32434"
        )
    is_junction = getattr(path, "is_junction", None)
    if path.is_symlink() or (callable(is_junction) and is_junction()):
        raise ValueError("training checkpoint must not be a link")
    if not path.is_file():
        raise FileNotFoundError(f"training checkpoint not found: {path}")
    try:
        with path.open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("training checkpoint must be a regular file")
            value = torch.load(handle, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, ValueError) as error:
        raise ValueError("training checkpoint is not a safe tensor payload") from error
    if not isinstance(value, dict):
        raise ValueError("training checkpoint payload must be a mapping")
    stack = [value]
    seen_containers: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (bool, int, float, str, bytes)):
            continue
        if isinstance(current, torch.Tensor):
            continue
        if isinstance(current, dict):
            identity = id(current)
            if identity in seen_containers:
                raise ValueError("training checkpoint contains a recursive mapping")
            seen_containers.add(identity)
            for key, item in current.items():
                if not isinstance(key, (str, int)) or isinstance(key, bool):
                    raise ValueError("training checkpoint mapping keys are invalid")
                stack.append(item)
            continue
        if isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in seen_containers:
                raise ValueError("training checkpoint contains a recursive sequence")
            seen_containers.add(identity)
            stack.extend(current)
            continue
        raise ValueError(
            f"training checkpoint contains an unsafe value: {type(current).__name__}"
        )
    return value


def _serialized_numpy_rng_state() -> dict[str, Any]:
    import torch

    algorithm, values, position, has_gauss, cached = np.random.get_state()
    values_array = np.asarray(values, dtype=np.uint32)
    return {
        "algorithm": str(algorithm),
        "values": torch.tensor(
            values_array.astype(np.int64, copy=True), dtype=torch.int64
        ),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _validated_numpy_rng_state(
    torch_module: Any, state: Mapping[str, Any]
) -> tuple[str, np.ndarray, int, int, float]:
    numpy_state = state.get("numpy_rng_state")
    if not isinstance(numpy_state, dict) or set(numpy_state) != {
        "algorithm",
        "values",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("training checkpoint NumPy RNG state is invalid")
    algorithm = numpy_state["algorithm"]
    values = numpy_state["values"]
    position = numpy_state["position"]
    has_gauss = numpy_state["has_gauss"]
    cached = numpy_state["cached_gaussian"]
    if (
        algorithm != "MT19937"
        or not isinstance(values, torch_module.Tensor)
        or values.dtype != torch_module.int64
        or values.ndim != 1
        or values.numel() != 624
        or isinstance(position, bool)
        or not isinstance(position, int)
        or not 0 <= position <= 624
        or isinstance(has_gauss, bool)
        or not isinstance(has_gauss, int)
        or has_gauss not in {0, 1}
        or isinstance(cached, bool)
        or not isinstance(cached, (int, float))
        or not math.isfinite(float(cached))
    ):
        raise ValueError("training checkpoint NumPy RNG state is invalid")
    cpu_values = values.detach().cpu()
    if bool(torch_module.any(cpu_values < 0)) or bool(
        torch_module.any(cpu_values > 2**32 - 1)
    ):
        raise ValueError("training checkpoint NumPy RNG state is invalid")
    restored = cpu_values.numpy().astype(np.uint32, copy=True)
    candidate = (algorithm, restored, position, has_gauss, float(cached))
    try:
        probe = np.random.RandomState()
        probe.set_state(candidate)
    except (TypeError, ValueError) as error:
        raise ValueError("training checkpoint NumPy RNG state is invalid") from error
    return candidate


def _validate_training_rng(
    torch_module: Any, state: Mapping[str, Any], device_type: str
) -> tuple[str, np.ndarray, int, int, float]:
    try:
        python_probe = random.Random()
        python_probe.setstate(state["python_rng_state"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("training checkpoint Python RNG state is invalid") from error

    torch_state = state.get("torch_rng_state")
    if (
        not isinstance(torch_state, torch_module.Tensor)
        or torch_state.dtype != torch_module.uint8
        or torch_state.ndim != 1
    ):
        raise ValueError("training checkpoint Torch RNG state is invalid")
    try:
        torch_module.Generator(device="cpu").set_state(torch_state.detach().cpu())
    except RuntimeError as error:
        raise ValueError("training checkpoint Torch RNG state is invalid") from error

    cuda_states = state.get("torch_cuda_rng_states")
    if not isinstance(cuda_states, list) or any(
        not isinstance(value, torch_module.Tensor)
        or value.dtype != torch_module.uint8
        or value.ndim != 1
        for value in cuda_states
    ):
        raise ValueError("training checkpoint CUDA RNG states are invalid")
    if device_type != "cuda" and cuda_states:
        raise ValueError("non-CUDA checkpoint contains CUDA RNG states")
    if device_type == "cuda":
        if (
            not torch_module.cuda.is_available()
            or len(cuda_states) != torch_module.cuda.device_count()
        ):
            raise ValueError("CUDA device count does not match the training checkpoint")
        try:
            for index, value in enumerate(cuda_states):
                torch_module.Generator(device=f"cuda:{index}").set_state(
                    value.detach().cpu()
                )
        except RuntimeError as error:
            raise ValueError("training checkpoint CUDA RNG states are invalid") from error
    return _validated_numpy_rng_state(torch_module, state)


def _model_state_fingerprint(model: Any | None) -> str | None:
    import torch

    if model is None:
        return None
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous().clone()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def _array_fingerprint(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    shape = list(array.shape)
    dtype = str(array.dtype)
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    chunk_bytes = 8 * 1024 * 1024
    if array.flags.c_contiguous:
        view = memoryview(array).cast("B")
        for start in range(0, len(view), chunk_bytes):
            digest.update(view[start : start + chunk_bytes])
    elif array.ndim == 0:
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    elif array.shape[0] > 0:
        bytes_per_row = max(int(array.nbytes // array.shape[0]), 1)
        rows_per_chunk = max(chunk_bytes // bytes_per_row, 1)
        for start in range(0, array.shape[0], rows_per_chunk):
            chunk = np.ascontiguousarray(array[start : start + rows_per_chunk])
            digest.update(memoryview(chunk).cast("B"))
    return {
        "shape": shape,
        "dtype": dtype,
        "sha256": digest.hexdigest(),
    }


def _training_signature(
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    weights: np.ndarray,
    teacher_model: Any | None,
    resume_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe every stable input that gives checkpoint tensors their meaning."""

    ignored_training_keys = {"checkpoint_every_epochs", "epochs", "resume_from"}
    signature = {
        "array_training_algorithm": "bitguard.array-neural.v2",
        "experiment_seed": int(config["experiment"]["seed"]),
        "dataset": config.get("dataset", {}),
        "split": config.get("split", {}),
        "preprocess": config.get("preprocess", {}),
        "model": config.get("model", {}),
        "loss": config.get("loss", {}),
        "training": {
            key: value
            for key, value in config.get("training", {}).items()
            if key not in ignored_training_keys
        },
        "arrays": {
            "x_train": _array_fingerprint(x_train),
            "y_train": _array_fingerprint(y_train),
            "x_validation": _array_fingerprint(x_validation),
            "y_validation": _array_fingerprint(y_validation),
            "class_weights": _array_fingerprint(weights),
        },
        "teacher_fingerprint": _model_state_fingerprint(teacher_model),
        "resume_context": resume_context or {},
    }
    return json.loads(json.dumps(signature, sort_keys=True, default=str))


def neural_train_step(
    model: Any,
    features: Any,
    target: Any,
    objective: Any,
    optimizer: Any,
    scaler: Any,
    config: dict[str, Any],
    teacher_model: Any | None,
) -> dict[str, float]:
    """Apply one neural optimizer update through the shared scientific boundary."""

    import torch

    from .models import clamp_binary_master_weights

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=features.device.type,
        enabled=scaler.is_enabled(),
    ):
        logits = model(features)
        with torch.no_grad():
            teacher_logits = None if teacher_model is None else teacher_model(features)
        output = objective(model, logits, target, teacher_logits)
    scaler.scale(output.total).backward()
    scaler.unscale_(optimizer)
    clip = float(config["training"].get("gradient_clip", 0.0))
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(optimizer)
    scaler.update()
    clamp_binary_master_weights(model)
    return {
        "loss": float(output.total.detach()),
        "detection": float(output.detection.detach()),
        "feature_cost": float(output.feature_cost.detach()),
        "fn": float(output.false_negative.detach()),
        "fp": float(output.false_positive.detach()),
    }


def neural_validation_metrics(
    y_validation: np.ndarray,
    validation_probability: np.ndarray,
    training_config: Mapping[str, Any],
) -> dict[str, float]:
    """Compute the shared complete-validation metrics and selection score."""

    validation_prediction = validation_probability.argmax(axis=1)
    class_indices = list(range(validation_probability.shape[1]))
    macro_f1 = float(
        f1_score(
            y_validation,
            validation_prediction,
            labels=class_indices,
            average="macro",
            zero_division=0,
        )
    )
    per_class_auprc: list[float] = []
    for class_index in class_indices:
        binary_target = (y_validation == class_index).astype(np.int8)
        if binary_target.min() == binary_target.max():
            per_class_auprc.append(0.0)
        else:
            per_class_auprc.append(
                float(
                    average_precision_score(
                        binary_target, validation_probability[:, class_index]
                    )
                )
            )
    macro_auprc = float(np.mean(per_class_auprc))
    attack_mask = y_validation != 0
    attack_recall = (
        float(np.mean(validation_prediction[attack_mask] != 0))
        if attack_mask.any()
        else 0.0
    )
    selection_weights = _validated_selection_weights(
        training_config["selection_weights"]
    )
    selection_score = (
        float(selection_weights["macro_f1"]) * macro_f1
        + float(selection_weights["macro_auprc"]) * macro_auprc
        + float(selection_weights["attack_recall"]) * attack_recall
    )
    if not math.isfinite(selection_score) or not 0.0 <= selection_score <= 1.0:
        raise ValueError("validation selection score must be finite in [0, 1]")
    return {
        "validation_macro_f1": macro_f1,
        "validation_macro_auprc": macro_auprc,
        "validation_attack_recall": attack_recall,
        "validation_selection_score": selection_score,
    }


def _validated_model_state(
    torch_module: Any,
    candidate: Any,
    reference: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(candidate, dict) or set(candidate) != set(reference):
        raise ValueError(f"training checkpoint {label} fields are invalid")
    normalized: dict[str, Any] = {}
    for key, expected in reference.items():
        value = candidate[key]
        if (
            not isinstance(value, torch_module.Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise ValueError(f"training checkpoint {label} tensor is invalid")
        normalized[key] = value.detach().cpu().clone()
    return normalized


def _replayed_best_state(
    records: list[dict[str, float | int]],
) -> tuple[float, int, int]:
    best_metric = -math.inf
    best_epoch = -1
    stale = 0
    for record in records:
        score = float(record["validation_selection_score"])
        if score > best_metric + 1e-6:
            best_metric = score
            best_epoch = int(record["epoch"])
            stale = 0
        else:
            stale += 1
    return best_metric, best_epoch, stale


def _validated_array_training_state(
    torch_module: Any,
    state: Mapping[str, Any],
    *,
    model: Any,
    target_epochs: int,
    device_type: str,
) -> tuple[
    int,
    list[dict[str, float | int]],
    dict[str, Any],
    float,
    int,
    int,
]:
    epoch = state["epoch"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= target_epochs
    ):
        raise ValueError("training checkpoint epoch is invalid")
    history = state["history"]
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("training checkpoint history is not a completed epoch prefix")
    records: list[dict[str, float | int]] = []
    for expected_epoch, candidate in enumerate(history, start=1):
        if not isinstance(candidate, dict) or set(candidate) != _NEURAL_HISTORY_FIELDS:
            raise ValueError("training checkpoint history record fields are invalid")
        record_epoch = candidate["epoch"]
        if (
            isinstance(record_epoch, bool)
            or not isinstance(record_epoch, int)
            or record_epoch != expected_epoch
        ):
            raise ValueError("training checkpoint history epoch is invalid")
        record: dict[str, float | int] = {"epoch": record_epoch}
        for name in _NEURAL_HISTORY_FIELDS - {"epoch"}:
            value = candidate[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("training checkpoint history metric is invalid")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError("training checkpoint history metric is non-finite")
            record[name] = normalized
        if float(record["learning_rate"]) < 0.0:
            raise ValueError("training checkpoint learning rate is invalid")
        if any(
            not 0.0 <= float(record[name]) <= 1.0
            for name in _NEURAL_VALIDATION_FIELDS
        ):
            raise ValueError("training checkpoint validation metric is out of range")
        records.append(record)

    best_metric = state["best_metric"]
    best_epoch = state["best_epoch"]
    stale = state["stale_epochs"]
    if (
        isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not math.isfinite(float(best_metric))
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or isinstance(stale, bool)
        or not isinstance(stale, int)
        or stale < 0
    ):
        raise ValueError("training checkpoint best/stale state is invalid")
    expected_metric, expected_best_epoch, expected_stale = _replayed_best_state(records)
    if (
        best_epoch != expected_best_epoch
        or float(best_metric) != expected_metric
        or stale != expected_stale
    ):
        raise ValueError("training checkpoint best/stale state is inconsistent")

    scheduler_state = state["scheduler_state_dict"]
    if not isinstance(scheduler_state, dict):
        raise ValueError("training checkpoint scheduler state is invalid")
    scheduler_epoch = scheduler_state.get("last_epoch")
    if (
        isinstance(scheduler_epoch, bool)
        or not isinstance(scheduler_epoch, int)
        or scheduler_epoch != epoch
    ):
        raise ValueError("training checkpoint scheduler phase is inconsistent")

    generator_state = state["generator_state"]
    if (
        not isinstance(generator_state, torch_module.Tensor)
        or generator_state.dtype != torch_module.uint8
        or generator_state.ndim != 1
    ):
        raise ValueError("training checkpoint data generator state is invalid")
    try:
        torch_module.Generator(device="cpu").set_state(generator_state.detach().cpu())
    except RuntimeError as error:
        raise ValueError("training checkpoint data generator state is invalid") from error
    _validate_training_rng(torch_module, state, device_type)

    reference = model.state_dict()
    _validated_model_state(
        torch_module, state["model_state_dict"], reference, "model state"
    )
    best_state = _validated_model_state(
        torch_module, state["best_state_dict"], reference, "best model state"
    )
    if not isinstance(state["optimizer_state_dict"], dict):
        raise ValueError("training checkpoint optimizer state is invalid")
    if not isinstance(state["scaler_state_dict"], dict):
        raise ValueError("training checkpoint scaler state is invalid")
    return epoch, records, best_state, float(best_metric), best_epoch, stale


def _fit_neural(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
    teacher_model: Any | None = None,
    *,
    checkpoint_path: Path | None = None,
    progress_path: Path | None = None,
    resume_from: Path | None = None,
    stop_after_epoch: int | None = None,
    resume_context: dict[str, Any] | None = None,
) -> NeuralFitResult:
    import torch
    from torch.utils.data import DataLoader, RandomSampler, TensorDataset

    from .losses import BitGuardObjective

    training_cfg = config["training"]
    device = _select_device(str(training_cfg.get("device", "auto")))
    model = model.to(device)
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
    weight_tensor = (
        torch.from_numpy(weights).to(device)
        if config["loss"].get("class_weighted", True)
        else None
    )
    objective = BitGuardObjective(config, weight_tensor, benign_index=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    batch_size = min(int(training_cfg["batch_size"]), len(x_train))
    seed = int(config["experiment"]["seed"])
    generator = torch.Generator().manual_seed(seed)
    worker_generator = torch.Generator().manual_seed(seed ^ 0xA5A5A5A5)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    sampler = RandomSampler(dataset, generator=generator)
    drop_last = len(dataset) > batch_size and len(dataset) % batch_size == 1
    num_workers = int(training_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=worker_generator,
        persistent_workers=num_workers > 0,
    )
    amp_enabled = bool(training_cfg.get("amp", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(torch, device.type, amp_enabled)
    patience = int(training_cfg["patience"])
    best_metric = -math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    records: list[dict[str, float | int]] = []
    start_epoch = 1
    training_signature = _training_signature(
        config,
        x_train,
        y_train,
        x_validation,
        y_validation,
        weights,
        teacher_model,
        resume_context,
    )
    if resume_from is not None:
        if not resume_from.exists():
            raise FileNotFoundError(f"training resume checkpoint not found: {resume_from}")
        state = _safe_torch_load(resume_from, device)
        required = {
            "format_version",
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "history",
            "best_state_dict",
            "best_metric",
            "best_epoch",
            "stale_epochs",
            "generator_state",
            "target_epochs",
            "device_type",
            "torch_rng_state",
            "torch_cuda_rng_states",
            "numpy_rng_state",
            "python_rng_state",
            "training_signature",
        }
        if set(state) != required:
            raise ValueError("training resume checkpoint fields are invalid")
        if (
            isinstance(state["format_version"], bool)
            or not isinstance(state["format_version"], int)
            or state["format_version"] != 3
        ):
            raise ValueError("unsupported array training checkpoint format")
        if (
            isinstance(state["target_epochs"], bool)
            or not isinstance(state["target_epochs"], int)
            or state["target_epochs"] != epochs
        ):
            raise ValueError(
                "training.epochs must match the checkpoint target; resume an interrupted "
                "run with its original total epoch count"
            )
        if state["device_type"] != device.type:
            raise ValueError("training resume checkpoint device type does not match this run")
        if state["training_signature"] != training_signature:
            raise ValueError(
                "training resume checkpoint does not match the current model, data split, "
                "preprocessing, loss, or optimizer configuration"
            )
        (
            completed_epoch,
            records,
            best_state,
            best_metric,
            best_epoch,
            stale,
        ) = _validated_array_training_state(
            torch,
            state,
            model=model,
            target_epochs=epochs,
            device_type=device.type,
        )
        try:
            model.load_state_dict(state["model_state_dict"])
        except RuntimeError as error:
            raise ValueError("training resume checkpoint does not match the model") from error
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        generator.set_state(state["generator_state"].cpu())
        _restore_training_rng(torch, state)
        start_epoch = completed_epoch + 1
    checkpoint_interval = max(int(training_cfg.get("checkpoint_every_epochs", 1)), 1)
    run_end_epoch = epochs if stop_after_epoch is None else min(int(stop_after_epoch), epochs)
    if run_end_epoch < start_epoch - 1:
        raise ValueError("stop_after_epoch precedes the resume checkpoint epoch")
    should_stop = bool(records) and stale >= patience
    try:
        for epoch in range(start_epoch, run_end_epoch + 1):
            if should_stop:
                break
            model.train()
            totals = {
                name: 0.0
                for name in ("loss", "detection", "feature_cost", "fn", "fp")
            }
            seen = 0
            for features, target in loader:
                features = features.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                step_metrics = neural_train_step(
                    model=model,
                    features=features,
                    target=target,
                    objective=objective,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    teacher_model=teacher_model,
                )
                batch_rows = len(features)
                seen += batch_rows
                for name, value in step_metrics.items():
                    totals[name] += value * batch_rows
            scheduler.step()
            validation_probability = _predict_neural_probabilities(
                model, x_validation, int(training_cfg["batch_size"]), device
            )
            validation_metrics = neural_validation_metrics(
                y_validation, validation_probability, training_cfg
            )
            validation_score = validation_metrics["validation_selection_score"]
            record: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **validation_metrics,
            }
            record.update(
                {
                    f"train_{key}": value / max(seen, 1)
                    for key, value in totals.items()
                }
            )
            records.append(record)
            should_stop = False
            if validation_score > best_metric + 1e-6:
                best_metric = validation_score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    should_stop = True
            if checkpoint_path is not None and (
                epoch % checkpoint_interval == 0
                or should_stop
                or epoch == run_end_epoch
            ):
                _save_training_state(
                    checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    records=records,
                    best_state=best_state,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    stale=stale,
                    generator=generator,
                    target_epochs=epochs,
                    device_type=device.type,
                    training_signature=training_signature,
                )
            if progress_path is not None:
                _save_training_progress(progress_path, records)
            if should_stop:
                break
    finally:
        _shutdown_persistent_workers(loader)
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return NeuralFitResult(model, pd.DataFrame(records), best_metric, best_epoch)


def _shutdown_persistent_workers(loader: Any) -> None:
    """Release persistent DataLoader processes before returning from a fit."""

    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        _close_training_iterator(iterator)
        loader._iterator = None


def _close_training_iterator(iterator: Any) -> None:
    """Close a public stream, with one isolated PyTorch compatibility fallback."""

    close = getattr(iterator, "close", None)
    if callable(close):
        close()
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _make_grad_scaler(torch_module: Any, device_type: str, enabled: bool) -> Any:
    """Use the current AMP API while retaining the declared PyTorch 2.6 floor."""

    modern = getattr(getattr(torch_module, "amp", None), "GradScaler", None)
    if modern is not None:
        return modern("cuda" if device_type == "cuda" else "cpu", enabled=enabled)
    legacy = torch_module.cuda.amp.GradScaler
    return legacy(enabled=enabled and device_type == "cuda")


def _restore_training_rng(torch_module: Any, state: dict[str, Any]) -> None:
    import random

    numpy_state = _validate_training_rng(
        torch_module, state, str(state["device_type"])
    )
    random.setstate(state["python_rng_state"])
    np.random.set_state(numpy_state)
    torch_module.set_rng_state(state["torch_rng_state"].cpu())
    cuda_states = [value.cpu() for value in state["torch_cuda_rng_states"]]
    if cuda_states and torch_module.cuda.is_available():
        if len(cuda_states) != torch_module.cuda.device_count():
            raise ValueError("CUDA device count does not match the training checkpoint")
        torch_module.cuda.set_rng_state_all(cuda_states)


def _save_training_state(
    path: Path,
    *,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    records: list[dict[str, float | int]],
    best_state: dict[str, Any] | None,
    best_metric: float,
    best_epoch: int,
    stale: int,
    generator: Any,
    target_epochs: int,
    device_type: str,
    training_signature: dict[str, Any],
) -> None:
    import torch
    import random

    if best_state is None:
        raise RuntimeError("cannot checkpoint training before a best state exists")
    model_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    _atomic_torch_save(
        path,
        {
            "format_version": 3,
            "epoch": int(epoch),
            "target_epochs": int(target_epochs),
            "device_type": str(device_type),
            "training_signature": training_signature,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": list(records),
            "best_state_dict": best_state,
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "stale_epochs": int(stale),
            "generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": (
                [value.cpu() for value in torch.cuda.get_rng_state_all()]
                if device_type == "cuda" and torch.cuda.is_available()
                else []
            ),
            "numpy_rng_state": _serialized_numpy_rng_state(),
            "python_rng_state": random.getstate(),
        },
    )


def _save_training_progress(
    path: Path, records: list[dict[str, float | int]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def _predict_neural_probabilities(
    model: Any,
    values: np.ndarray,
    batch_size: int,
    device: Any | None = None,
) -> np.ndarray:
    import torch

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            outputs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _build_neural(
    config: dict[str, Any],
    preprocessor: FeaturePreprocessor,
    output_dim: int,
    *,
    input_indices: np.ndarray | None = None,
    hidden_dims: list[int] | None = None,
    force_bnn: bool = False,
) -> Any:
    from .models import build_model

    if input_indices is None:
        input_groups = preprocessor.input_groups
        input_dim = preprocessor.encoded_dimension
        costs = preprocessor.feature_costs
    else:
        all_groups = preprocessor.input_groups[input_indices]
        unique = sorted(set(int(item) for item in all_groups))
        remap = {old: new for new, old in enumerate(unique)}
        input_groups = np.asarray([remap[int(item)] for item in all_groups], dtype=np.int64)
        input_dim = len(input_indices)
        assert preprocessor.feature_costs is not None
        costs = preprocessor.feature_costs[unique]
    assert costs is not None
    return build_model(
        config,
        input_dim,
        output_dim,
        input_groups,
        costs,
        hidden_dims=hidden_dims,
        force_bnn=force_bnn,
    )


def _checkpoint(
    model: Any,
    config: dict[str, Any],
    preprocessor: FeaturePreprocessor,
    path: Path,
    output_labels: list[str],
    hidden_dims: list[int],
    input_indices: np.ndarray | None = None,
) -> None:
    import torch

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "model_type": config["model"]["type"],
            "input_dim": len(input_indices) if input_indices is not None else preprocessor.encoded_dimension,
            "output_dim": len(output_labels),
            "output_labels": output_labels,
            "hidden_dims": list(hidden_dims),
            "dropout": float(config["model"].get("dropout", 0.0)),
            "binary_first_layer": bool(config["model"].get("binary_first_layer", True)),
            "input_indices": input_indices,
            "input_groups": preprocessor.input_groups if input_indices is None else preprocessor.input_groups[input_indices],
            "feature_costs": preprocessor.feature_costs,
        },
        path,
    )


def _fit_classical(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model
    if model_type == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed)
        weights = class_weights(y_train, int(y_train.max()) + 1)[y_train]
        model.fit(x_train, y_train, sample_weight=weights)
        return model
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "install the optional dependency with: pip install -e '.[xgboost]'"
            ) from error
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=seed,
        )
        weights = class_weights(y_train, int(y_train.max()) + 1)[y_train]
        model.fit(x_train, y_train, sample_weight=weights)
        return model
    raise ValueError(f"unsupported classical model: {model_type}")


def _to_full_probabilities(
    preprocessor: FeaturePreprocessor,
    known_probabilities: np.ndarray,
    unencoded_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels, _, active_plus_unknown = preprocessor.apply_open_set(
        known_probabilities, unencoded_values
    )
    full = np.zeros((len(known_probabilities), len(CANONICAL_LABELS)), dtype=np.float32)
    for index, label in enumerate([*preprocessor.active_labels, "unknown_like"]):
        full[:, CANONICAL_LABELS.index(label)] = active_plus_unknown[:, index]
    full /= np.maximum(full.sum(axis=1, keepdims=True), 1e-12)
    return labels, full


def _load_and_split(config: dict[str, Any]) -> tuple[DataSplit, list[str]]:
    source = load_dataset(config)
    validate_labels(source.frame)
    if config["split"]["strategy"] != "cross":
        split = make_split(source, config)
        features = list(source.feature_columns)
        del source
        return split, features
    cross_path = config["dataset"].get("cross_path")
    cross_type = config["dataset"].get("cross_type")
    if not cross_path or not cross_type:
        raise ValueError("cross split requires dataset.cross_path and dataset.cross_type")
    target_config = copy.deepcopy(config)
    target_config["dataset"]["type"] = cross_type
    target_config["dataset"]["path"] = cross_path
    target = load_dataset(target_config)
    validate_labels(target.frame)
    split, shared = make_cross_split(source, target, config)
    del source, target
    return split, shared


def run_training(config_path: str | Path) -> Path:
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    run_dir = create_run_dir(config)
    save_yaml(config, run_dir / "resolved_config.yaml")
    save_json(environment_manifest(), run_dir / "environment.json")
    split, candidate_features = _load_and_split(config)
    save_json(split.manifest, run_dir / "split_manifest.json")
    preprocessor = FeaturePreprocessor(config).fit(split.train, candidate_features)
    preprocessor.calibrate_open_set(split.validation)
    preprocessor.save(run_dir / "preprocessor.joblib")
    save_json(preprocessor.feature_manifest(), run_dir / "feature_manifest.json")

    x_train = preprocessor.transform(split.train)
    x_validation = preprocessor.transform(split.validation)
    x_validation_raw = preprocessor.transform_unencoded(split.validation)
    y_train = preprocessor.encode_labels(split.train)
    y_validation = preprocessor.encode_labels(split.validation)
    if np.any(y_validation < 0):
        raise ValueError("validation contains a class absent from training; move it to held-out test")
    resume_context = {
        "selected_features": list(preprocessor.selected_features),
        "active_labels": list(preprocessor.active_labels),
        "input_groups": preprocessor.input_groups.tolist(),
        "split_manifest": split.manifest,
    }
    attack_prior = np.zeros(len(CANONICAL_LABELS), dtype=np.float64)
    train_attack_counts = split.train.loc[
        split.train["behavior_label"] != "benign", "behavior_label"
    ].value_counts()
    for label, count in train_attack_counts.items():
        if label in CANONICAL_LABELS and label != "unknown_like":
            attack_prior[CANONICAL_LABELS.index(label)] = float(count)
    split.train = pd.DataFrame()

    model_type = str(config["model"]["type"])
    history = pd.DataFrame()
    model_summary: dict[str, Any]
    if model_type in {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
    }:
        model = _fit_classical(model_type, x_train, y_train, seed)
        validation_known = model.predict_proba(x_validation).astype(np.float32)
        joblib.dump(model, run_dir / "best_model.joblib")
        model_summary = {
            "model_type": model_type,
            "artifact": "best_model.joblib",
            "classes": preprocessor.active_labels,
            **_process_resource_summary(run_dir / "best_model.joblib"),
        }
    else:
        from .models import feature_gate_summary, parameter_summary

        teacher_model = None
        if float(config["loss"].get("distillation_alpha", 0.0)) > 0:
            if model_type == "fp32_mlp":
                raise ValueError("distillation_alpha requires a BNN student, not fp32_mlp")
            teacher_config = copy.deepcopy(config)
            teacher_config["model"]["type"] = "fp32_mlp"
            teacher_config["loss"]["distillation_alpha"] = 0.0
            teacher_config["loss"]["lambda_feature"] = 0.0
            teacher_model = _build_neural(
                teacher_config, preprocessor, len(preprocessor.active_labels)
            )
            teacher_fit = _fit_neural(
                teacher_model,
                x_train,
                y_train,
                x_validation,
                y_validation,
                class_weights(y_train, len(preprocessor.active_labels)),
                teacher_config,
                checkpoint_path=run_dir / "teacher_training_state.pt",
                progress_path=run_dir / "teacher_training_history.partial.csv",
                resume_context={**resume_context, "training_role": "teacher"},
            )
            teacher_model = teacher_fit.model
            teacher_fit.history.to_csv(run_dir / "teacher_training_history.csv", index=False)
            _checkpoint(
                teacher_model,
                teacher_config,
                preprocessor,
                run_dir / "teacher_model.pt",
                preprocessor.active_labels,
                list(config["model"]["hidden_dims"]),
            )
        model = _build_neural(config, preprocessor, len(preprocessor.active_labels))
        resume_value = config["training"].get("resume_from")
        resume_path = resolve_path(config, resume_value) if resume_value else None
        fit = _fit_neural(
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
            class_weights(y_train, len(preprocessor.active_labels)),
            config,
            teacher_model,
            checkpoint_path=run_dir / "last_training_state.pt",
            progress_path=run_dir / "training_history.partial.csv",
            resume_from=resume_path,
            resume_context={**resume_context, "training_role": "main"},
        )
        model = fit.model
        history = fit.history
        history.to_csv(run_dir / "training_history.csv", index=False)
        validation_known = _predict_neural_probabilities(
            model, x_validation, int(config["training"]["batch_size"])
        )
        _checkpoint(
            model,
            config,
            preprocessor,
            run_dir / "best_model.pt",
            preprocessor.active_labels,
            list(config["model"]["hidden_dims"]),
        )
        model_summary = {
            "model_type": model_type,
            "artifact": "best_model.pt",
            "classes": preprocessor.active_labels,
            "best_validation_selection_score": fit.best_validation_score,
            "best_epoch": fit.best_epoch,
            **parameter_summary(model),
            **feature_gate_summary(model),
            **_process_resource_summary(run_dir / "best_model.pt"),
        }
        import torch

        sample = torch.from_numpy(x_validation[:1]).to(next(model.parameters()).device)
        model_summary["latency"] = benchmark_torch_model(
            model,
            sample,
            int(config["evaluation"]["benchmark_warmup"]),
            int(config["evaluation"]["benchmark_repeats"]),
        )

    preprocessor.calibrate_confidence_threshold(validation_known, x_validation_raw)
    preprocessor.save(run_dir / "preprocessor.joblib")
    save_json(preprocessor.feature_manifest(), run_dir / "feature_manifest.json")
    save_yaml(preprocessor.config, run_dir / "calibrated_config.yaml")
    validation_true = split.validation["behavior_label"].astype(str).to_numpy()
    _, validation_full = _to_full_probabilities(
        preprocessor, validation_known, x_validation_raw
    )
    tiny_model: Any | None = None
    tiny_indices: np.ndarray | None = None
    calibration: Any | None = None
    boolean_calibration: Any | None = None
    main_ops = 0
    tiny_ops = 0
    cascade_results: dict[str, Any] | None = None

    if bool(config["cascade"].get("enabled", False)):
        if model_type in {
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "xgboost",
        }:
            raise ValueError("cascade currently requires a neural Main model")
        tiny_budget = min(
            int(config["cascade"]["tiny_feature_budget"]), len(preprocessor.selected_features)
        )
        tiny_indices = preprocessor.encoder.encoded_indices_for_first(
            tiny_budget, len(preprocessor.selected_features)
        )
        tiny_config = copy.deepcopy(config)
        tiny_config["model"]["type"] = "vanilla_bnn"
        tiny_config["loss"]["distillation_alpha"] = 0.0
        tiny_model = _build_neural(
            tiny_config,
            preprocessor,
            2,
            input_indices=tiny_indices,
            hidden_dims=list(config["cascade"]["hidden_dims"]),
            force_bnn=True,
        )
        y_train_tiny = (y_train != 0).astype(np.int64)
        y_validation_tiny = (y_validation != 0).astype(np.int64)
        tiny_fit = _fit_neural(
            tiny_model,
            x_train[:, tiny_indices],
            y_train_tiny,
            x_validation[:, tiny_indices],
            y_validation_tiny,
            class_weights(y_train_tiny, 2),
            tiny_config,
            checkpoint_path=run_dir / "tiny_training_state.pt",
            progress_path=run_dir / "tiny_training_history.partial.csv",
            resume_context={
                **resume_context,
                "training_role": "tiny",
                "tiny_indices": tiny_indices.tolist(),
            },
        )
        tiny_model = tiny_fit.model
        tiny_fit.history.to_csv(run_dir / "tiny_training_history.csv", index=False)
        tiny_validation = _predict_neural_probabilities(
            tiny_model,
            x_validation[:, tiny_indices],
            int(config["training"]["batch_size"]),
        )
        calibration = tune_exit_threshold(
            tiny_validation[:, 0],
            split.validation["behavior_label"].to_numpy(),
            float(config["cascade"]["min_attack_recall"]),
            int(config["cascade"]["threshold_grid_size"]),
            float(config["cascade"]["false_negative_cost"]),
        )
        save_json(calibration.to_dict(), run_dir / "cascade_calibration.json")
        if bool(config["cascade"].get("boolean_fast_path_enabled", True)):
            boolean_calibration = tune_boolean_fast_path(
                split.validation,
                list(config["cascade"].get("boolean_fast_path_features", [])),
                float(config["cascade"]["min_attack_recall"]),
            )
        else:
            from .cascade import BooleanFastPathCalibration

            boolean_calibration = BooleanFastPathCalibration(False, [], {}, 1.0, 0.0)
        save_json(boolean_calibration.to_dict(), run_dir / "boolean_fast_path.json")
        boolean_validation = apply_boolean_fast_path(
            split.validation, boolean_calibration.upper_thresholds
        )
        validation_full, _, _ = route_with_temporal_state(
            split.validation[
                [
                    column
                    for column in (
                        "source_file",
                        "device_id",
                        "timestamp",
                        "sequence_index",
                        "row_uid",
                    )
                    if column in split.validation
                ]
            ],
            tiny_validation[:, 0],
            validation_full,
            CANONICAL_LABELS,
            calibration,
            config,
            attack_prior,
            boolean_validation,
        )
        _checkpoint(
            tiny_model,
            tiny_config,
            preprocessor,
            run_dir / "tiny_model.pt",
            ["benign", "attack"],
            list(config["cascade"]["hidden_dims"]),
            tiny_indices,
        )
        main_ops = estimate_dense_operations(
            preprocessor.encoded_dimension,
            list(config["model"]["hidden_dims"]),
            len(preprocessor.active_labels),
        )
        tiny_ops = estimate_dense_operations(
            len(tiny_indices), list(config["cascade"]["hidden_dims"]), 2
        )
    fixed_fpr_thresholds = calibrate_fixed_fpr_thresholds(
        validation_true,
        CANONICAL_LABELS,
        validation_full,
        config["evaluation"].get("fixed_fpr_targets", (1e-2, 1e-3)),
    )
    del validation_true, validation_full, validation_known, x_validation_raw
    del x_train, y_train, x_validation, y_validation
    split.validation = pd.DataFrame()

    x_test = preprocessor.transform(split.test)
    x_test_raw = preprocessor.transform_unencoded(split.test)
    if model_type in {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
    }:
        test_known = model.predict_proba(x_test).astype(np.float32)
    else:
        test_known = _predict_neural_probabilities(
            model, x_test, int(config["training"]["batch_size"])
        )
    main_labels, test_full = _to_full_probabilities(preprocessor, test_known, x_test_raw)
    exit_stage = np.full(len(split.test), 2, dtype=np.int8)

    if tiny_model is not None:
        assert tiny_indices is not None
        assert calibration is not None
        assert boolean_calibration is not None
        boolean_test = apply_boolean_fast_path(
            split.test, boolean_calibration.upper_thresholds
        )
        tiny_test = _predict_neural_probabilities(
            tiny_model,
            x_test[:, tiny_indices],
            int(config["training"]["batch_size"]),
        )
        test_full, exit_stage, routing_summary = route_with_temporal_state(
            split.test[
                [
                    column
                    for column in (
                        "source_file",
                        "device_id",
                        "timestamp",
                        "sequence_index",
                        "row_uid",
                    )
                    if column in split.test
                ]
            ],
            tiny_test[:, 0],
            test_full,
            CANONICAL_LABELS,
            calibration,
            config,
            attack_prior,
            boolean_test,
        )
        main_labels = np.where(exit_stage < 2, "benign", main_labels).astype(str)
        true_evaluation = np.where(
            split.test["behavior_label"].isin(preprocessor.active_labels),
            split.test["behavior_label"],
            "unknown_like",
        )
        attack_mask = true_evaluation != "benign"
        cascade_results = {
            "calibration": calibration.to_dict(),
            "boolean_fast_path": boolean_calibration.to_dict(),
            "test_routing": routing_summary,
            "test_attack_escalation_recall": (
                float(np.mean(exit_stage[attack_mask] == 2)) if attack_mask.any() else None
            ),
            **cascade_operation_summary(
                exit_stage,
                tiny_ops,
                main_ops,
                len(boolean_calibration.features) if boolean_calibration.enabled else 0,
            ),
        }
    del x_test, x_test_raw, test_known

    original_true = split.test["behavior_label"].astype(str).to_numpy()
    evaluation_true = np.where(
        np.isin(original_true, preprocessor.active_labels), original_true, "unknown_like"
    ).astype(str)
    metadata_columns = [
        column
        for column in (
            "row_uid",
            "dataset",
            "source_file",
            "sequence_index",
            "device_id",
            "raw_attack",
            "timestamp",
        )
        if column in split.test
    ]
    predictions = split.test[metadata_columns].copy()
    predictions["original_true_label"] = original_true
    predictions["true_label"] = evaluation_true
    predictions["predicted_label"] = main_labels
    predictions["exit_stage"] = exit_stage
    predictions["has_wall_clock_time"] = bool(
        split.manifest.get("provenance", split.manifest.get("target_provenance", {})).get(
            "has_wall_clock_time", False
        )
    )
    predictions["temporal_continuity"] = bool(split.manifest.get("temporal_continuity", False))
    for index, label in enumerate(CANONICAL_LABELS):
        predictions[f"prob_{label}"] = test_full[:, index]
    metrics = classification_metrics(
        evaluation_true,
        main_labels,
        CANONICAL_LABELS,
        test_full,
        list(config["evaluation"]["high_risk_labels"]),
        operating_thresholds=fixed_fpr_thresholds,
    )
    metrics["fixed_fpr"]["score_pipeline"] = (
        "cascade" if cascade_results is not None else "main_open_set"
    )
    confusion_frame(evaluation_true, main_labels).to_csv(run_dir / "confusion_matrix.csv")
    if bool(config["evaluation"].get("save_predictions", True)):
        predictions.to_csv(run_dir / "predictions.csv", index=False)
    plot_files: list[str] = []
    if bool(config["evaluation"].get("make_plots", True)):
        plot_files = make_plots(predictions, CANONICAL_LABELS, run_dir)
    operational: dict[str, Any] | None = None
    if bool(config["temporal"].get("enabled", False)):
        temporal_predictions, operational = replay_predictions(predictions, config)
        temporal_predictions.to_csv(run_dir / "temporal_predictions.csv", index=False)
        save_json(operational, run_dir / "operational_metrics.json")
    result = {
        "classification": metrics,
        "model": model_summary,
        "cascade": cascade_results,
        "operational": operational,
        "plots": plot_files,
        "research_validity": {
            "unknown_test_labels": "Any behavior absent from train is evaluated as unknown_like.",
            "native_cross_dataset_padding": False,
            "automatic_network_action": False,
            "pytorch_bnn_speed_claim": False,
        },
    }
    save_json(model_summary, run_dir / "model_summary.json")
    save_json(result, run_dir / "metrics.json")
    return run_dir
