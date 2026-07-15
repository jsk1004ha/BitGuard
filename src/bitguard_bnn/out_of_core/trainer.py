"""Exact, step-resumable neural training over verified Parquet batches."""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from bitguard_bnn.out_of_core.cache import class_weights_from_counts
from bitguard_bnn.out_of_core.dataset import (
    DATASET_ALGORITHM,
    DataCursor,
    ParquetTrainingDataset,
    iter_ordered_batches,
)
from bitguard_bnn.out_of_core.manifest import stable_fingerprint
from bitguard_bnn.out_of_core.prepare import verify_prepared_dataset
from bitguard_bnn.trainer import (
    NeuralFitResult,
    _close_training_iterator,
    _make_grad_scaler,
    _restore_training_rng,
    _save_training_progress,
    _select_device,
    neural_train_step,
)


STREAMING_CHECKPOINT_FORMAT = 3
_METRIC_NAMES = ("loss", "detection", "feature_cost", "fn", "fp")
_VALIDATION_NAMES = (
    "validation_macro_f1",
    "validation_macro_auprc",
    "validation_attack_recall",
)

ValidationCallback = Callable[[Any, Any], Mapping[str, float]]


class StreamingTrainingInterrupted(RuntimeError):
    """Intentional test/runtime interruption after a durable step checkpoint."""


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _streaming_signature(
    dataset: ParquetTrainingDataset,
    config: Mapping[str, Any],
    counts: Mapping[str, int],
    active_labels: Sequence[str],
    *,
    training_role: str,
    feature_indices: tuple[int, ...] | None,
    binary_attack_target: bool,
    teacher_fingerprint: str | None,
) -> dict[str, Any]:
    prepared = verify_prepared_dataset(dataset.descriptor_path)
    if (
        str(prepared.shard_fingerprint) != str(dataset.manifest_fingerprint)
        or str(prepared.preprocessing_fingerprint)
        != str(dataset.preprocessing_fingerprint)
    ):
        raise RuntimeError("training dataset no longer matches its prepared descriptor")
    descriptor = prepared.to_dict()
    training = dict(config.get("training", {}))
    for ignored in (
        "checkpoint_every_epochs",
        "checkpoint_every_steps",
        "resume_from",
        "num_workers",
        "device",
        "epochs",
        "batch_size",
        "shuffle_buffer_rows",
    ):
        training.pop(ignored, None)
    signature = {
        "prepared_descriptor_fingerprint": descriptor["fingerprint"],
        "preparation_fingerprint": prepared.preparation_fingerprint,
        "shard_fingerprint": prepared.shard_fingerprint,
        "preprocessor_fingerprint": prepared.preprocessing_fingerprint,
        "normalized_source_fingerprint": prepared.normalized_source_fingerprint,
        "split_fingerprint": prepared.split_fingerprint,
        "split": dataset.split,
        "dataset_algorithm": DATASET_ALGORITHM,
        "batch_size": dataset.batch_size,
        "seed": dataset.seed,
        "shuffle_buffer_rows": dataset.shuffle_buffer_rows,
        "experiment_seed": int(config["experiment"]["seed"]),
        "model": config.get("model", {}),
        "loss": config.get("loss", {}),
        "optimizer_and_selection": training,
        "class_counts": {label: int(counts[label]) for label in active_labels},
        "active_labels": list(active_labels),
        "training_transform": {
            "role": training_role,
            "feature_indices": None if feature_indices is None else list(feature_indices),
            "target": "benign-versus-attack" if binary_attack_target else "active-label-index",
            "teacher_fingerprint": teacher_fingerprint,
        },
    }
    canonical = _canonical(signature)
    canonical["fingerprint"] = stable_fingerprint(canonical)
    return canonical


def _validate_inputs(
    dataset: ParquetTrainingDataset,
    config: Mapping[str, Any],
    counts: Mapping[str, int],
    active_labels: Sequence[str],
    *,
    training_role: str,
    feature_indices: Sequence[int] | None,
    binary_attack_target: bool,
) -> tuple[tuple[str, ...], tuple[int, ...] | None, np.ndarray]:
    labels = tuple(active_labels)
    if not labels or labels[0] != "benign":
        raise ValueError("active_labels must start with benign")
    if dataset.split != "train":
        raise ValueError("fit_neural_streaming requires the prepared train split")
    training = config["training"]
    if int(training["batch_size"]) != dataset.batch_size:
        raise ValueError("training.batch_size does not match the Parquet dataset")
    if int(config["experiment"]["seed"]) != dataset.seed:
        raise ValueError("experiment.seed does not match the Parquet dataset")
    if int(training["shuffle_buffer_rows"]) != dataset.shuffle_buffer_rows:
        raise ValueError("training.shuffle_buffer_rows does not match the Parquet dataset")
    if not isinstance(training_role, str) or not training_role or "\x00" in training_role:
        raise ValueError("training_role must be a non-empty string")
    indices: tuple[int, ...] | None
    if feature_indices is None:
        indices = None
    else:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in feature_indices):
            raise ValueError("feature_indices must contain non-negative integers")
        indices = tuple(feature_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("feature_indices must be non-empty and unique")
    if not isinstance(binary_attack_target, bool):
        raise ValueError("binary_attack_target must be boolean")
    if binary_attack_target:
        binary_counts = {
            "benign": int(counts[labels[0]]),
            "attack": sum(int(counts[label]) for label in labels[1:]),
        }
        weights = class_weights_from_counts(binary_counts, ("benign", "attack"))
    else:
        weights = class_weights_from_counts(counts, labels)
    return labels, indices, weights


def _validation_record(
    callback: ValidationCallback,
    model: Any,
    device: Any,
    selection_weights: Mapping[str, float],
) -> dict[str, float]:
    model.eval()
    values = dict(callback(model, device))
    missing = sorted(set(_VALIDATION_NAMES) - set(values))
    if missing:
        raise ValueError(f"validation callback is missing metrics: {missing}")
    metrics = {name: float(values[name]) for name in _VALIDATION_NAMES}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("validation callback returned a non-finite metric")
    metrics["validation_selection_score"] = (
        float(selection_weights["macro_f1"]) * metrics["validation_macro_f1"]
        + float(selection_weights["macro_auprc"])
        * metrics["validation_macro_auprc"]
        + float(selection_weights["attack_recall"])
        * metrics["validation_attack_recall"]
    )
    return metrics


def _cpu_state(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _model_fingerprint(model: Any | None) -> str | None:
    if model is None:
        return None
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_streaming_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    cursor: DataCursor,
    epoch_phase: str,
    partial_totals: Mapping[str, float],
    partial_seen_rows: int,
    records: list[dict[str, float | int]],
    best_state: Mapping[str, Any] | None,
    best_metric: float,
    best_epoch: int,
    stale: int,
    target_epochs: int,
    device_type: str,
    scientific_signature: Mapping[str, Any],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "format_version": STREAMING_CHECKPOINT_FORMAT,
        "cursor": cursor,
        "global_optimizer_step": cursor.optimizer_step,
        "epoch_phase": epoch_phase,
        "partial_totals": {name: float(partial_totals[name]) for name in _METRIC_NAMES},
        "partial_seen_rows": int(partial_seen_rows),
        "target_epochs": int(target_epochs),
        "device_type": str(device_type),
        "scientific_signature": dict(scientific_signature),
        "model_state_dict": _cpu_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": list(records),
        "best_state_dict": (
            None
            if best_state is None
            else {key: value.detach().cpu().clone() for key, value in best_state.items()}
        ),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "stale_epochs": int(stale),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            [value.cpu() for value in torch.cuda.get_rng_state_all()]
            if device_type == "cuda" and torch.cuda.is_available()
            else []
        ),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    try:
        torch.save(state, temporary)
        _fsync_path(temporary)
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_streaming_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    target_epochs: int,
    device: Any,
    scientific_signature: Mapping[str, Any],
) -> tuple[
    DataCursor,
    str,
    dict[str, float],
    int,
    list[dict[str, float | int]],
    dict[str, Any] | None,
    float,
    int,
    int,
]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"streaming training checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    required = {
        "format_version",
        "cursor",
        "global_optimizer_step",
        "epoch_phase",
        "partial_totals",
        "partial_seen_rows",
        "target_epochs",
        "device_type",
        "scientific_signature",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "history",
        "best_state_dict",
        "best_metric",
        "best_epoch",
        "stale_epochs",
        "torch_rng_state",
        "torch_cuda_rng_states",
        "numpy_rng_state",
        "python_rng_state",
    }
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError("streaming checkpoint fields are invalid")
    if state["format_version"] != STREAMING_CHECKPOINT_FORMAT:
        raise ValueError("unsupported streaming checkpoint format")
    if state["scientific_signature"] != scientific_signature:
        raise ValueError("streaming checkpoint scientific signature mismatch")
    if int(state["target_epochs"]) != target_epochs:
        raise ValueError("training.epochs must match the streaming checkpoint target")
    if str(state["device_type"]) != device.type:
        raise ValueError("streaming checkpoint device type mismatch")
    cursor = state["cursor"]
    if not isinstance(cursor, DataCursor):
        raise ValueError("streaming checkpoint cursor is invalid")
    if int(state["global_optimizer_step"]) != cursor.optimizer_step:
        raise ValueError("streaming checkpoint optimizer step and cursor mismatch")
    phase = state["epoch_phase"]
    if phase not in {"training", "validation", "epoch_boundary"}:
        raise ValueError("streaming checkpoint epoch phase is invalid")
    records = list(state["history"])
    expected_epochs = list(range(1, len(records) + 1))
    if [record.get("epoch") for record in records] != expected_epochs:
        raise ValueError("streaming checkpoint history is not a completed epoch prefix")
    partial = state["partial_totals"]
    if not isinstance(partial, Mapping) or set(partial) != set(_METRIC_NAMES):
        raise ValueError("streaming checkpoint partial totals are invalid")
    totals = {name: float(partial[name]) for name in _METRIC_NAMES}
    if any(not math.isfinite(value) for value in totals.values()):
        raise ValueError("streaming checkpoint partial totals are non-finite")
    seen = state["partial_seen_rows"]
    if isinstance(seen, bool) or not isinstance(seen, int) or seen < 0:
        raise ValueError("streaming checkpoint partial row count is invalid")
    if phase in {"training", "validation"}:
        if cursor.epoch != len(records) + 1 or seen <= 0:
            raise ValueError("streaming checkpoint active epoch cursor is inconsistent")
    elif (
        cursor.epoch != len(records) + 1
        or cursor.shard_position != 0
        or cursor.batch_position != 0
        or seen != 0
        or any(value != 0.0 for value in totals.values())
    ):
        raise ValueError("streaming checkpoint epoch-boundary cursor is inconsistent")
    if cursor.epoch < 1 or cursor.epoch > target_epochs + 1:
        raise ValueError("streaming checkpoint cursor epoch is out of range")
    best_raw = state["best_state_dict"]
    if best_raw is None:
        if records:
            raise ValueError("completed history requires a best model state")
        best_state = None
    elif isinstance(best_raw, Mapping):
        best_state = {
            key: value.detach().cpu().clone() for key, value in best_raw.items()
        }
    else:
        raise ValueError("streaming checkpoint best model state is invalid")
    try:
        model.load_state_dict(state["model_state_dict"])
    except RuntimeError as error:
        raise ValueError("streaming checkpoint does not match the model") from error
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    expected_scheduler_epoch = cursor.epoch if phase == "validation" else cursor.epoch - 1
    if int(scheduler.last_epoch) != expected_scheduler_epoch:
        raise ValueError("streaming checkpoint scheduler phase is inconsistent")
    scaler.load_state_dict(state["scaler_state_dict"])
    _restore_training_rng(torch, state)
    return (
        cursor,
        str(phase),
        totals,
        seen,
        records,
        best_state,
        float(state["best_metric"]),
        int(state["best_epoch"]),
        int(state["stale_epochs"]),
    )


def fit_neural_streaming(
    model: Any,
    dataset: ParquetTrainingDataset,
    class_counts: Mapping[str, int],
    active_labels: Sequence[str],
    config: dict[str, Any],
    validation_callback: ValidationCallback,
    teacher_model: Any | None = None,
    *,
    checkpoint_path: Path | None = None,
    progress_path: Path | None = None,
    resume_from: Path | None = None,
    training_role: str = "main",
    feature_indices: Sequence[int] | None = None,
    binary_attack_target: bool = False,
    stop_after_optimizer_step: int | None = None,
) -> NeuralFitResult:
    """Train without materializing prepared rows and resume at the next unapplied batch."""

    import torch

    from bitguard_bnn.losses import BitGuardObjective

    labels, indices, weights = _validate_inputs(
        dataset,
        config,
        class_counts,
        active_labels,
        training_role=training_role,
        feature_indices=feature_indices,
        binary_attack_target=binary_attack_target,
    )
    del labels
    training = config["training"]
    epochs = int(training["epochs"])
    device = _select_device(str(training.get("device", "auto")))
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
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    amp_enabled = bool(training.get("amp", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(torch, device.type, amp_enabled)
    signature = _streaming_signature(
        dataset,
        config,
        class_counts,
        active_labels,
        training_role=training_role,
        feature_indices=indices,
        binary_attack_target=binary_attack_target,
        teacher_fingerprint=_model_fingerprint(teacher_model),
    )
    cursor = DataCursor(epoch=1, shard_position=0, batch_position=0, optimizer_step=0)
    epoch_phase = "training"
    totals = {name: 0.0 for name in _METRIC_NAMES}
    seen = 0
    records: list[dict[str, float | int]] = []
    best_state: dict[str, Any] | None = None
    best_metric = -math.inf
    best_epoch = -1
    stale = 0
    if resume_from is not None:
        (
            cursor,
            epoch_phase,
            totals,
            seen,
            records,
            best_state,
            best_metric,
            best_epoch,
            stale,
        ) = _load_streaming_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            target_epochs=epochs,
            device=device,
            scientific_signature=signature,
        )
    if stop_after_optimizer_step is not None and (
        isinstance(stop_after_optimizer_step, bool)
        or not isinstance(stop_after_optimizer_step, int)
        or stop_after_optimizer_step <= cursor.optimizer_step
    ):
        raise ValueError("stop_after_optimizer_step must exceed the current optimizer step")
    checkpoint_interval = int(training["checkpoint_every_steps"])
    patience = int(training["patience"])
    num_workers = int(training.get("num_workers", 0))
    should_stop = bool(records) and stale >= patience
    if epoch_phase == "epoch_boundary":
        epoch_phase = "training"

    while cursor.epoch <= epochs and not should_stop:
        epoch = cursor.epoch
        if epoch_phase == "training":
            dataset.set_epoch(epoch, cursor)
            model.train()
            iterator = iter(iter_ordered_batches(dataset, num_workers=num_workers))
        elif epoch_phase == "validation":
            iterator = iter(())
        else:
            raise RuntimeError("streaming training entered an invalid epoch phase")
        try:
            for batch in iterator:
                if batch.get("cursor") != cursor:
                    raise RuntimeError("stream batch cursor does not match the next unapplied batch")
                features = np.asarray(batch["features"], dtype=np.float32)
                if indices is not None:
                    if indices and max(indices) >= features.shape[1]:
                        raise ValueError("feature_indices exceed the prepared feature dimension")
                    features = features[:, indices]
                targets = np.asarray(batch["labels"], dtype=np.int64)
                if binary_attack_target:
                    targets = (targets != 0).astype(np.int64, copy=False)
                feature_tensor = torch.from_numpy(features).to(device, non_blocking=True)
                target_tensor = torch.from_numpy(targets).to(device, non_blocking=True)
                step_metrics = neural_train_step(
                    model=model,
                    features=feature_tensor,
                    target=target_tensor,
                    objective=objective,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    teacher_model=teacher_model,
                )
                rows = len(feature_tensor)
                if rows <= 0:
                    raise RuntimeError("streaming training received an empty batch")
                seen += rows
                for name in _METRIC_NAMES:
                    totals[name] += float(step_metrics[name]) * rows
                next_cursor = batch.get("next_cursor")
                if not isinstance(next_cursor, DataCursor):
                    raise RuntimeError("streaming batch next cursor is invalid")
                if next_cursor.optimizer_step != cursor.optimizer_step + 1:
                    raise RuntimeError("streaming batch optimizer step is not contiguous")
                cursor = next_cursor
                if checkpoint_path is not None and cursor.optimizer_step % checkpoint_interval == 0:
                    _save_streaming_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        cursor=cursor,
                        epoch_phase="training",
                        partial_totals=totals,
                        partial_seen_rows=seen,
                        records=records,
                        best_state=best_state,
                        best_metric=best_metric,
                        best_epoch=best_epoch,
                        stale=stale,
                        target_epochs=epochs,
                        device_type=device.type,
                        scientific_signature=signature,
                    )
                if (
                    stop_after_optimizer_step is not None
                    and cursor.optimizer_step == stop_after_optimizer_step
                ):
                    if checkpoint_path is None:
                        raise ValueError(
                            "stop_after_optimizer_step requires checkpoint_path"
                        )
                    if cursor.optimizer_step % checkpoint_interval != 0:
                        _save_streaming_checkpoint(
                            checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            cursor=cursor,
                            epoch_phase="training",
                            partial_totals=totals,
                            partial_seen_rows=seen,
                            records=records,
                            best_state=best_state,
                            best_metric=best_metric,
                            best_epoch=best_epoch,
                            stale=stale,
                            target_epochs=epochs,
                            device_type=device.type,
                            scientific_signature=signature,
                        )
                    raise StreamingTrainingInterrupted(
                        f"streaming training stopped after optimizer step {cursor.optimizer_step}"
                    )
        finally:
            _close_training_iterator(iterator)
        if cursor.shard_position != len(dataset.entries) or cursor.batch_position != 0:
            raise RuntimeError("streaming epoch ended before exact prepared-row coverage")
        if seen != dataset.row_count:
            raise RuntimeError("streaming partial row coverage does not match the train split")
        if epoch_phase == "training":
            scheduler.step()
            epoch_phase = "validation"
            if checkpoint_path is not None:
                _save_streaming_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    cursor=cursor,
                    epoch_phase=epoch_phase,
                    partial_totals=totals,
                    partial_seen_rows=seen,
                    records=records,
                    best_state=best_state,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    stale=stale,
                    target_epochs=epochs,
                    device_type=device.type,
                    scientific_signature=signature,
                )
        validation = _validation_record(
            validation_callback,
            model,
            device,
            training["selection_weights"],
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **validation,
        }
        record.update({f"train_{name}": totals[name] / max(seen, 1) for name in _METRIC_NAMES})
        records.append(record)
        score = validation["validation_selection_score"]
        if score > best_metric + 1e-6:
            best_metric = score
            best_epoch = epoch
            best_state = _cpu_state(model)
            stale = 0
        else:
            stale += 1
            should_stop = stale >= patience
        cursor = DataCursor(
            epoch=epoch + 1,
            shard_position=0,
            batch_position=0,
            optimizer_step=cursor.optimizer_step,
        )
        totals = {name: 0.0 for name in _METRIC_NAMES}
        seen = 0
        if checkpoint_path is not None:
            _save_streaming_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                cursor=cursor,
                epoch_phase="epoch_boundary",
                partial_totals=totals,
                partial_seen_rows=seen,
                records=records,
                best_state=best_state,
                best_metric=best_metric,
                best_epoch=best_epoch,
                stale=stale,
                target_epochs=epochs,
                device_type=device.type,
                scientific_signature=signature,
            )
        epoch_phase = "training"
        if progress_path is not None:
            _save_training_progress(progress_path, records)
    if best_state is None:
        raise RuntimeError("streaming training did not produce a best state")
    model.load_state_dict(best_state)
    return NeuralFitResult(model, pd.DataFrame(records), best_metric, best_epoch)


__all__ = [
    "STREAMING_CHECKPOINT_FORMAT",
    "StreamingTrainingInterrupted",
    "fit_neural_streaming",
]
