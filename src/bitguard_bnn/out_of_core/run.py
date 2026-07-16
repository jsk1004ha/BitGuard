"""Memory-bounded orchestration for verified prepared Parquet datasets."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, cast

import numpy as np
import pandas as pd

from bitguard_bnn.cascade import CascadeCalibration, CascadeStreamRouter
from bitguard_bnn.config import (
    create_run_dir,
    environment_manifest,
    load_config,
    resolve_path,
    save_json,
    save_yaml,
    seed_everything,
)
from bitguard_bnn.constants import CANONICAL_LABELS
from bitguard_bnn.export import export_run
from bitguard_bnn.out_of_core.dataset import ParquetTrainingDataset
from bitguard_bnn.out_of_core.evaluate import evaluate_prediction_batches
from bitguard_bnn.out_of_core.manifest import stable_fingerprint
from bitguard_bnn.out_of_core.manifest import read_split_manifest
from bitguard_bnn.out_of_core.metrics import StreamingClassificationMetrics
from bitguard_bnn.out_of_core.prepare import PreparedDataset, verify_prepared_dataset
from bitguard_bnn.preprocess import FeaturePreprocessor


_EXACT_IN_MEMORY_CLASSICAL_MODELS = frozenset(
    {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
    }
)

_TEST_PREDICTION_ORDER_ALGORITHM = {
    "wall_clock": ("timestamp", "device_id", "row_uid", "storage_position"),
    "sequence_fallback": (
        "device_id",
        "sequence_index",
        "row_uid",
        "storage_position",
    ),
    "timestamp_policy": "wall_clock_only_if_all_rows_are_finite",
    "restore": ("storage_position",),
}

# Resource collection is diagnostic, so a production phase must not pay for a
# recursive work-tree scan dozens of times per second.
_PHASE_SAMPLE_INTERVAL_SECONDS = 1.0


def _add_exception_note(error: BaseException, note: str) -> None:
    """Attach secondary failure detail without requiring Python 3.11."""

    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


def _reject_unsupported_model(config: Mapping[str, Any]) -> None:
    model_type = str(config["model"]["type"])
    if model_type in _EXACT_IN_MEMORY_CLASSICAL_MODELS:
        raise ValueError(
            f"{model_type} does not support exact out-of-core fitting; choose a neural "
            "model or an explicitly supported incremental baseline"
        )


def _scientific_training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select caller-visible settings that the prepared config must pin exactly."""

    dataset = dict(config["dataset"])
    for locator in (
        "root",
        "shard_manifest",
        "prepared_descriptor",
        "source_manifest",
        "schema_report",
    ):
        dataset.pop(locator, None)
    return {
        "experiment_seed": config["experiment"]["seed"],
        "dataset": dataset,
        "preprocess": copy.deepcopy(config["preprocess"]),
        "model": copy.deepcopy(config["model"]),
        "loss": copy.deepcopy(config["loss"]),
        "training": copy.deepcopy(config["training"]),
        "cascade": copy.deepcopy(config["cascade"]),
        "temporal": copy.deepcopy(config["temporal"]),
        "evaluation": copy.deepcopy(config["evaluation"]),
    }


def _require_matching_scientific_contract(
    caller: Mapping[str, Any], verified: Mapping[str, Any]
) -> None:
    caller_contract = _scientific_training_contract(caller)
    verified_contract = _scientific_training_contract(verified)
    if stable_fingerprint(caller_contract) != stable_fingerprint(verified_contract):
        raise ValueError(
            "caller config scientific training contract does not match the verified "
            "prepared config; only experiment.output_dir may be overridden"
        )


def _descriptor_candidates(config_path: Path, config: Mapping[str, Any]) -> list[Path]:
    dataset = config["dataset"]
    explicit = dataset.get("prepared_descriptor")
    if isinstance(explicit, str) and explicit.strip():
        resolved_explicit = resolve_path(dict(config), explicit)
        if resolved_explicit is None:
            raise ValueError("dataset.prepared_descriptor could not be resolved")
        return [resolved_explicit]

    manifest = resolve_path(dict(config), dataset.get("shard_manifest"))
    if manifest is None:
        raise ValueError("dataset.shard_manifest is required for prepared discovery")
    dataset_name = str(dataset.get("type", "")).casefold()
    candidates = [
        manifest.parent / "prepared_dataset.json",
        manifest.parent / "descriptor.json",
        config_path.with_suffix(".prepared.json"),
    ]
    prepared_root = manifest.parent.parent
    if prepared_root.name.casefold() == "prepared":
        control = prepared_root.parent / ".bitguard" / "prepared" / dataset_name
        if control.is_dir():
            candidates.extend(sorted(control.glob("*.json"), reverse=True))
    return candidates


def _resolve_prepared_descriptor(
    config_path: Path, config: Mapping[str, Any]
) -> PreparedDataset:
    errors: list[str] = []
    for candidate in _descriptor_candidates(config_path, config):
        if not candidate.is_file():
            continue
        try:
            prepared = verify_prepared_dataset(candidate)
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"{candidate}: {error}")
            continue
        resolved = Path(prepared.resolved_config_path).resolve()
        template = Path(prepared.template_config_path).resolve()
        if config_path.resolve() in {resolved, template}:
            return prepared
    detail = f" ({'; '.join(errors)})" if errors else ""
    raise FileNotFoundError(
        "unable to locate a verified prepared dataset descriptor; set "
        f"dataset.prepared_descriptor or run through bootstrap{detail}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_bytes(path: Path, *, cancelled: threading.Event | None = None) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        for item in path.rglob("*"):
            if cancelled is not None and cancelled.is_set():
                break
            try:
                observed = item.stat()
            except OSError:
                continue
            if stat.S_ISREG(observed.st_mode):
                total += int(observed.st_size)
    except OSError:
        # Phase work trees are actively compacted; a directory can disappear
        # between rglob's scan and descent without invalidating the sample.
        pass
    return total


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        getrusage = getattr(resource, "getrusage")
        rusage_self = getattr(resource, "RUSAGE_SELF")
        value = int(getrusage(rusage_self).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except ImportError:
        return None


def _current_rss_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            if not psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            ):
                return None
            return int(counters.WorkingSetSize)
        except (AttributeError, OSError, ValueError):
            return None
    try:
        page_count = int(Path("/proc/self/statm").read_text().split()[1])
        return page_count * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _phase_resource_sample(
    temporary_directory: Path, *, cancelled: threading.Event | None = None
) -> tuple[int | None, int]:
    """Return one bounded phase-resource sample through a patchable test seam."""

    return _current_rss_bytes(), _directory_bytes(
        temporary_directory, cancelled=cancelled
    )


@contextmanager
def _phase(
    records: list[dict[str, Any]], name: str, temporary_directory: Path
) -> Iterator[None]:
    started = time.perf_counter()
    rss_start, disk_before = _phase_resource_sample(temporary_directory)
    sampled_peak = [rss_start]
    sampled_disk_peak = [disk_before]
    stopped = threading.Event()
    sampling_errors: list[BaseException] = []

    def sample_resources() -> None:
        try:
            while not stopped.wait(_PHASE_SAMPLE_INTERVAL_SECONDS):
                rss_value, disk_value = _phase_resource_sample(
                    temporary_directory, cancelled=stopped
                )
                if rss_value is not None and (
                    sampled_peak[0] is None or rss_value > sampled_peak[0]
                ):
                    sampled_peak[0] = rss_value
                sampled_disk_peak[0] = max(sampled_disk_peak[0], disk_value)
        except BaseException as error:
            sampling_errors.append(error)
            stopped.set()

    sampler = threading.Thread(
        target=sample_resources,
        name=f"bitguard-phase-sampler-{name}",
        daemon=True,
    )
    sampler.start()
    primary_error: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        stopped.set()
        sampler.join()
        if sampling_errors:
            sampling_error = sampling_errors[0]
            if primary_error is None:
                raise RuntimeError(
                    f"phase resource sampling failed for {name}"
                ) from sampling_error
            _add_exception_note(
                primary_error,
                f"phase resource sampler also failed for {name}: {sampling_error!r}",
            )
        else:
            final_sample_failed = False
            try:
                rss_end, disk_end = _phase_resource_sample(temporary_directory)
                lifetime_peak = _peak_rss_bytes()
            except BaseException as sampling_error:
                if primary_error is None:
                    raise RuntimeError(
                        f"phase resource sampling failed for {name}"
                    ) from sampling_error
                _add_exception_note(
                    primary_error,
                    f"phase final resource diagnostics also failed for {name}: "
                    f"{sampling_error!r}",
                )
                final_sample_failed = True
            if (
                not final_sample_failed
                and rss_end is not None
                and (sampled_peak[0] is None or rss_end > sampled_peak[0])
            ):
                sampled_peak[0] = rss_end
            if not final_sample_failed:
                sampled_disk_peak[0] = max(sampled_disk_peak[0], disk_end)
                records.append(
                    {
                        "phase": name,
                        "elapsed_seconds": time.perf_counter() - started,
                        "rss_measurement": "sampled_process_working_set_1s",
                        "process_rss_bytes_at_start": rss_start,
                        "process_rss_bytes_at_end": rss_end,
                        "phase_sampled_peak_rss_bytes": sampled_peak[0],
                        "process_lifetime_peak_rss_bytes_at_phase_end": lifetime_peak,
                        "temporary_disk_bytes_before": disk_before,
                        "temporary_disk_bytes_after": disk_end,
                        "temporary_disk_measurement": "sampled_directory_size_1s",
                        "temporary_disk_sampled_peak_bytes": sampled_disk_peak[0],
                    }
                )


@contextmanager
def _temporary_calibration_cache(root: Path, layout: Any) -> Iterator[Any]:
    """Own a calibration cache only for one phase and remove it on every exit."""

    from bitguard_bnn.out_of_core.cache import CalibrationCache

    preexisting = os.path.lexists(root)
    cache: Any | None = None
    primary_error: BaseException | None = None
    try:
        cache = CalibrationCache.create(root, layout)
        yield cache
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        try:
            if cache is not None:
                cache.close()
        except BaseException as error:
            cleanup_error = error
        if not preexisting and os.path.lexists(root):
            try:
                shutil.rmtree(root, ignore_errors=False)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    _add_exception_note(
                        cleanup_error, f"cache directory cleanup also failed: {error}"
                    )
        if primary_error is not None and cleanup_error is not None:
            _add_exception_note(
                primary_error, f"validation cache cleanup also failed: {cleanup_error}"
            )
        elif cleanup_error is not None:
            raise cleanup_error


def _class_counts(prepared: PreparedDataset, labels: list[str]) -> dict[str, int]:
    manifest = json.loads(
        Path(prepared.shard_manifest_path).read_text(encoding="utf-8")
    )
    counts = {label: 0 for label in labels}
    for entry in manifest["entries"]:
        if entry.get("split") == "train" and str(entry.get("label")) in counts:
            counts[str(entry["label"])] += int(entry["rows"])
    if (
        any(value <= 0 for value in counts.values())
        or sum(counts.values()) != prepared.train_count
    ):
        raise RuntimeError("prepared train class counts do not match active labels")
    return counts


def _dataset(
    prepared: PreparedDataset, config: Mapping[str, Any], split: str
) -> ParquetTrainingDataset:
    training = config["training"]
    return ParquetTrainingDataset(
        prepared.descriptor_path,
        split=split,
        batch_size=int(training["batch_size"]),
        seed=int(config["experiment"]["seed"]),
        shuffle_buffer_rows=int(training["shuffle_buffer_rows"]),
    )


def _iter_evaluation_batches(
    dataset: ParquetTrainingDataset,
) -> Iterator[dict[str, Any]]:
    """Slice deterministic source chunks to the configured inference batch size."""

    dataset.set_epoch(0)
    seen = 0
    for source in dataset:
        rows = len(source["row_uid"])
        if rows <= 0:
            raise RuntimeError("evaluation dataset emitted an empty source chunk")
        for group in ("metadata", "boolean_raw"):
            values = source.get(group)
            if not isinstance(values, Mapping):
                raise RuntimeError(f"evaluation batch has invalid {group}")
            if any(len(value) != rows for value in values.values()):
                raise RuntimeError(f"evaluation batch has misaligned {group} rows")
        for name in ("features", "unencoded", "labels", "row_uid"):
            if name in source and len(source[name]) != rows:
                raise RuntimeError(f"evaluation batch has misaligned {name} rows")
        for start in range(0, rows, dataset.batch_size):
            stop = min(start + dataset.batch_size, rows)
            batch = dict(source)
            for name in ("features", "unencoded", "labels", "row_uid"):
                if name in source:
                    batch[name] = source[name][start:stop]
            batch["metadata"] = {
                name: value[start:stop] for name, value in source["metadata"].items()
            }
            batch["boolean_raw"] = {
                name: value[start:stop] for name, value in source["boolean_raw"].items()
            }
            seen += stop - start
            yield batch
    if seen != dataset.row_count:
        raise RuntimeError("evaluation batches did not cover the complete split")


def _validation_callback(
    dataset: ParquetTrainingDataset,
    labels: list[str],
    temporary_directory: Path,
    *,
    feature_indices: np.ndarray | None = None,
    binary_attack_target: bool = False,
) -> Any:
    callback_labels = ["benign", "attack"] if binary_attack_target else labels
    high_risk = callback_labels[1:]

    def callback(model: Any, device: Any) -> dict[str, float]:
        import torch

        temporary_directory.mkdir(parents=True, exist_ok=True)
        metrics = StreamingClassificationMetrics(
            probability_labels=callback_labels,
            high_risk_labels=high_risk,
            temporary_directory=temporary_directory,
            score_run_rows=max(dataset.batch_size, 2),
        )
        try:
            seen = 0
            with torch.inference_mode():
                for batch in _iter_evaluation_batches(dataset):
                    values = np.asarray(batch["features"], dtype=np.float32)
                    if feature_indices is not None:
                        values = values[:, feature_indices]
                    probability = (
                        torch.softmax(model(torch.from_numpy(values).to(device)), dim=1)
                        .cpu()
                        .numpy()
                    )
                    raw_true = np.asarray(
                        batch["metadata"]["behavior_label"], dtype=str
                    )
                    true = (
                        np.where(raw_true == "benign", "benign", "attack")
                        if binary_attack_target
                        else raw_true
                    )
                    predicted = np.asarray(callback_labels, dtype=str)[
                        probability.argmax(axis=1)
                    ]
                    metrics.update(
                        true,
                        predicted,
                        probability,
                        np.asarray(batch["row_uid"], dtype=str),
                    )
                    seen += len(true)
            if seen != dataset.row_count:
                raise RuntimeError(
                    "validation callback did not consume the complete split"
                )
            result = metrics.finalize()
            attack_rate = result["high_risk_false_negative_rate"]
            return {
                "validation_macro_f1": float(result["macro_f1"]),
                "validation_macro_auprc": float(result["macro_auprc"] or 0.0),
                "validation_attack_recall": (
                    0.0 if attack_rate is None else 1.0 - float(attack_rate)
                ),
            }
        finally:
            metrics.cleanup()

    return callback


def _attack_prior(counts: Mapping[str, int]) -> np.ndarray:
    prior: np.ndarray = np.zeros(len(CANONICAL_LABELS), dtype=np.float64)
    for label, count in counts.items():
        if label != "benign" and label in CANONICAL_LABELS:
            prior[CANONICAL_LABELS.index(label)] = float(count)
    return prior


def _to_full_probabilities(
    preprocessor: FeaturePreprocessor,
    known: np.ndarray,
    unencoded: np.ndarray,
) -> np.ndarray:
    _labels, _unknown, active = preprocessor.apply_open_set(known, unencoded)
    full: np.ndarray = np.zeros((len(known), len(CANONICAL_LABELS)), dtype=np.float32)
    for column, label in enumerate([*preprocessor.active_labels, "unknown_like"]):
        full[:, CANONICAL_LABELS.index(label)] = active[:, column]
    return full / np.maximum(full.sum(axis=1, keepdims=True), 1e-12)


def _text_widths(dataset: ParquetTrainingDataset) -> tuple[int, int]:
    device_width = 1
    source_width = 1
    seen = 0
    for batch in _iter_evaluation_batches(dataset):
        metadata = batch["metadata"]
        device_width = max(
            device_width,
            *(len(str(value).encode("utf-8")) for value in metadata["device_id"]),
        )
        source_width = max(
            source_width,
            *(len(str(value).encode("utf-8")) for value in metadata["source_file"]),
        )
        seen += len(batch["row_uid"])
    if seen != dataset.row_count:
        raise RuntimeError("validation width scan did not consume the complete split")
    return device_width, source_width


def _prediction_temporal_contract(
    prepared: PreparedDataset, config: Mapping[str, Any]
) -> dict[str, bool]:
    """Derive replay claims only from verified complete-data artifacts.

    Prepared-dataset verification pins the schema report and split manifest
    fingerprints before this helper is reached.  The flags therefore describe
    the complete normalized source and its unsampled split, not properties
    guessed from whichever prediction batch happens to be observed first.
    """

    schema_report = json.loads(
        Path(prepared.schema_report_path).read_text(encoding="utf-8")
    )
    split_manifest = read_split_manifest(prepared.split_manifest_path)
    dataset_config = config["dataset"]
    dataset_name = str(prepared.dataset).casefold()
    time_column = str(dataset_config.get("time_column", "")).strip()
    files = schema_report.get("files", [])
    accepted_files = [
        value
        for value in files
        if isinstance(value, Mapping) and int(value.get("accepted_rows", 0)) > 0
    ]
    schema_has_time = bool(
        dataset_name == "botiot"
        and time_column
        and accepted_files
        and all(time_column in value.get("columns", []) for value in accepted_files)
        and int(schema_report.get("rejected_rows", -1)) == 0
        and int(schema_report.get("accepted_rows", -1)) == prepared.total_count
    )
    rejections = split_manifest.get("rejections", {})
    split_time_complete = bool(
        int(rejections.get("missing_timestamp", -1)) == 0
        and int(rejections.get("nonfinite_timestamp", -1)) == 0
    )
    has_wall_clock_time = schema_has_time and split_time_complete

    unsampled = all(
        dataset_config.get(name) is None
        for name in ("max_rows_per_file", "max_rows_per_class", "max_loaded_rows")
    )
    strategy = str(split_manifest.get("strategy", ""))
    counts = split_manifest.get("counts", {})
    membership = split_manifest.get("membership", {})
    complete_split = bool(
        int(split_manifest.get("inspection", {}).get("rows", -1))
        == prepared.total_count
        and int(membership.get("rows", -1)) == prepared.total_count
        and sum(
            int(counts.get(name, -prepared.total_count))
            for name in ("train", "validation", "test")
        )
        == prepared.total_count
    )
    temporal_continuity = bool(
        unsampled
        and complete_split
        and strategy in {"time", "device"}
        and (strategy != "time" or has_wall_clock_time)
    )
    return {
        "has_wall_clock_time": bool(has_wall_clock_time),
        "temporal_continuity": temporal_continuity,
    }


def _test_inference_contract(
    *,
    prepared: PreparedDataset,
    run_dir: Path,
    preprocessor: FeaturePreprocessor,
    runtime_config: Mapping[str, Any],
    tiny_indices: np.ndarray | None,
    calibration: CascadeCalibration,
    boolean_calibration: Any,
    fixed_fpr: Mapping[float, float],
    attack_prior: np.ndarray,
    prediction_metadata: Mapping[str, bool],
) -> dict[str, Any]:
    """Bind every persisted or configured input that can alter test inference."""

    feature_manifest = preprocessor.feature_manifest()
    tiny_path = run_dir / "tiny_model.pt"
    semantic: dict[str, Any] = {
        "algorithm": "bitguard.full-test-inference-contract.v2",
        "prepared": {
            "descriptor_fingerprint": prepared.to_dict()["fingerprint"],
            "split_fingerprint": prepared.split_fingerprint,
            "shard_fingerprint": prepared.shard_fingerprint,
            "normalized_source_fingerprint": prepared.normalized_source_fingerprint,
            "preprocessing_fingerprint": prepared.preprocessing_fingerprint,
        },
        "checkpoints": {
            "main_sha256": _sha256(run_dir / "best_model.pt"),
            "tiny_sha256": _sha256(tiny_path) if tiny_path.is_file() else None,
        },
        "preprocessor": {
            "artifact_sha256": _sha256(run_dir / "preprocessor.joblib"),
            "feature_manifest_sha256": _sha256(run_dir / "feature_manifest.json"),
            "calibrated_config_sha256": _sha256(run_dir / "calibrated_config.yaml"),
            "feature_manifest": feature_manifest,
            "preprocess_config": copy.deepcopy(preprocessor.config["preprocess"]),
            "selected_features": list(preprocessor.selected_features),
            "encoded_dimension": preprocessor.encoded_dimension,
        },
        "cascade": {
            "enabled": bool(runtime_config["cascade"].get("enabled", False)),
            "tiny_input_indices": (
                None
                if tiny_indices is None
                else np.asarray(tiny_indices, dtype=np.int64).tolist()
            ),
            "calibration": calibration.to_dict(),
            "config": copy.deepcopy(runtime_config["cascade"]),
        },
        "boolean_fast_path": {
            "calibration": boolean_calibration.to_dict(),
            "enabled_setting": bool(
                runtime_config["cascade"].get("boolean_fast_path_enabled", True)
            ),
            "available_features": list(boolean_calibration.features),
            "minimum_attack_recall": float(
                runtime_config["cascade"]["min_attack_recall"]
            ),
        },
        "routing": {
            "class_labels": list(CANONICAL_LABELS),
            "cascade_config": copy.deepcopy(runtime_config["cascade"]),
            "temporal_config": copy.deepcopy(runtime_config["temporal"]),
            "attack_prior": np.asarray(attack_prior, dtype=np.float64).tolist(),
            "order_algorithm": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in _TEST_PREDICTION_ORDER_ALGORITHM.items()
            },
        },
        "fixed_fpr": {
            "score_pipeline": "routed_attack_score=1-p_benign",
            "targets": [
                float(value)
                for value in runtime_config["evaluation"].get("fixed_fpr_targets", [])
            ],
            "thresholds": {
                str(key): float(value) for key, value in sorted(fixed_fpr.items())
            },
        },
        "prediction_metadata": {
            name: bool(prediction_metadata[name])
            for name in ("has_wall_clock_time", "temporal_continuity")
        },
    }
    return {
        "format_version": 2,
        "contract": semantic,
        "fingerprint": stable_fingerprint(semantic),
    }


def _boolean_mask(batch: Mapping[str, Any], calibration: Any) -> np.ndarray:
    rows = len(batch["row_uid"])
    if calibration is None or not calibration.enabled:
        return np.zeros(rows, dtype=bool)
    mask: np.ndarray = np.ones(rows, dtype=bool)
    for name, threshold in calibration.upper_thresholds.items():
        values = np.asarray(batch["boolean_raw"][name], dtype=np.float64)
        mask &= np.isfinite(values) & (values <= float(threshold))
    return mask


def _temporal_prediction_order(*, missing_timestamps: int) -> str:
    """Return the persisted inference contract's canonical temporal SQL order."""

    key = "wall_clock" if missing_timestamps == 0 else "sequence_fallback"
    return ", ".join(_TEST_PREDICTION_ORDER_ALGORITHM[key])


def _prediction_batches(
    dataset: ParquetTrainingDataset,
    main_model: Any,
    preprocessor: FeaturePreprocessor,
    *,
    tiny_model: Any | None,
    tiny_indices: np.ndarray | None,
    calibration: CascadeCalibration | None,
    boolean_calibration: Any | None,
    config: dict[str, Any],
    attack_prior: np.ndarray,
    prediction_metadata: Mapping[str, bool],
    temporary_directory: Path | None = None,
) -> Iterator[Mapping[str, Any]]:
    import torch

    device = next(main_model.parameters()).device
    router = (
        CascadeStreamRouter(list(CANONICAL_LABELS), calibration, config, attack_prior)
        if tiny_model is not None
        and tiny_indices is not None
        and calibration is not None
        else None
    )
    main_model.eval()
    if tiny_model is not None:
        tiny_model.eval()

    def infer(batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.asarray(batch["features"], dtype=np.float32)
        unencoded = np.asarray(batch["unencoded"], dtype=np.float32)
        known = (
            torch.softmax(main_model(torch.from_numpy(features).to(device)), dim=1)
            .cpu()
            .numpy()
        )
        main_probability = _to_full_probabilities(preprocessor, known, unencoded)
        tiny_probability: np.ndarray = np.zeros(len(features), dtype=np.float32)
        if tiny_model is not None and tiny_indices is not None:
            tiny_probability = (
                torch.softmax(
                    tiny_model(torch.from_numpy(features[:, tiny_indices]).to(device)),
                    dim=1,
                )[:, 0]
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        return (
            main_probability,
            tiny_probability,
            _boolean_mask(batch, boolean_calibration),
        )

    def output_batch(
        rows: list[tuple[Any, ...]], probabilities: np.ndarray, stages: np.ndarray
    ) -> Mapping[str, Any]:
        predicted = np.asarray(CANONICAL_LABELS, dtype=str)[
            probabilities.argmax(axis=1)
        ]
        count = len(rows)
        return {
            "row_uid": np.asarray([row[1] for row in rows], dtype=str),
            "true_label": np.asarray([row[2] for row in rows], dtype=str),
            "predicted_label": predicted,
            "probabilities": probabilities,
            "exit_stage": stages,
            "source_file": np.asarray([row[3] for row in rows], dtype=str),
            "device_id": np.asarray([row[4] for row in rows], dtype=str),
            "timestamp": np.asarray(
                [np.nan if row[5] is None else row[5] for row in rows],
                dtype=np.float64,
            ),
            "sequence_index": np.asarray([row[6] for row in rows], dtype=np.int64),
            "raw_attack": np.asarray([row[7] for row in rows], dtype=str),
            "has_wall_clock_time": np.full(
                count,
                bool(prediction_metadata["has_wall_clock_time"]),
                dtype=bool,
            ),
            "temporal_continuity": np.full(
                count,
                bool(prediction_metadata["temporal_continuity"]),
                dtype=bool,
            ),
        }

    seen = 0
    if router is None or not router.use_temporal:
        with torch.inference_mode():
            for batch in _iter_evaluation_batches(dataset):
                routed, tiny_probability, boolean = infer(batch)
                metadata_values = dict(batch["metadata"])
                metadata_values["row_uid"] = np.asarray(batch["row_uid"], dtype=str)
                stages: np.ndarray = np.full(len(routed), 2, dtype=np.int8)
                if router is not None:
                    routed, stages, _scores = router.route_batch(
                        pd.DataFrame(metadata_values),
                        tiny_probability,
                        routed,
                        boolean,
                    )
                rows = [
                    (
                        seen + offset,
                        str(batch["row_uid"][offset]),
                        str(batch["metadata"]["behavior_label"][offset]),
                        str(batch["metadata"]["source_file"][offset]),
                        str(batch["metadata"]["device_id"][offset]),
                        float(batch["metadata"]["timestamp"][offset]),
                        int(batch["metadata"]["sequence_index"][offset]),
                        str(batch["metadata"]["raw_attack"][offset]),
                    )
                    for offset in range(len(routed))
                ]
                seen += len(routed)
                yield output_batch(rows, routed, stages)
        if seen != dataset.row_count:
            raise RuntimeError("test evaluation did not consume the complete split")
        return

    if temporary_directory is None:
        raise ValueError("temporal prediction routing requires a temporary directory")
    temporary_directory.mkdir(parents=True, exist_ok=True)
    database_path = temporary_directory / "test-temporal-routing.sqlite3"
    if database_path.exists():
        raise RuntimeError(f"temporal prediction work already exists: {database_path}")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """
            CREATE TABLE prediction_rows (
                storage_position INTEGER PRIMARY KEY,
                row_uid TEXT NOT NULL,
                true_label TEXT NOT NULL,
                source_file TEXT NOT NULL,
                device_id TEXT NOT NULL,
                timestamp REAL,
                sequence_index INTEGER NOT NULL,
                raw_attack TEXT NOT NULL,
                main_probabilities BLOB NOT NULL,
                tiny_benign_probability REAL NOT NULL,
                boolean_fast_path INTEGER NOT NULL,
                routed_probabilities BLOB,
                exit_stage INTEGER
            )
            """
        )
        with torch.inference_mode():
            for batch in _iter_evaluation_batches(dataset):
                main_probability, tiny_probability, boolean = infer(batch)
                values: list[tuple[Any, ...]] = []
                for offset in range(len(main_probability)):
                    timestamp = float(batch["metadata"]["timestamp"][offset])
                    values.append(
                        (
                            seen + offset,
                            str(batch["row_uid"][offset]),
                            str(batch["metadata"]["behavior_label"][offset]),
                            str(batch["metadata"]["source_file"][offset]),
                            str(batch["metadata"]["device_id"][offset]),
                            timestamp if np.isfinite(timestamp) else None,
                            int(batch["metadata"]["sequence_index"][offset]),
                            str(batch["metadata"]["raw_attack"][offset]),
                            sqlite3.Binary(
                                np.asarray(
                                    main_probability[offset], dtype="<f4"
                                ).tobytes()
                            ),
                            float(tiny_probability[offset]),
                            int(boolean[offset]),
                        )
                    )
                connection.executemany(
                    """
                    INSERT INTO prediction_rows (
                        storage_position, row_uid, true_label, source_file,
                        device_id, timestamp, sequence_index, raw_attack,
                        main_probabilities, tiny_benign_probability,
                        boolean_fast_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                connection.commit()
                seen += len(main_probability)
        if seen != dataset.row_count:
            raise RuntimeError("test evaluation did not consume the complete split")

        missing_timestamps = int(
            connection.execute(
                "SELECT COUNT(*) FROM prediction_rows WHERE timestamp IS NULL"
            ).fetchone()[0]
        )
        order = _temporal_prediction_order(missing_timestamps=missing_timestamps)
        chronological = connection.execute(
            "SELECT storage_position, row_uid, true_label, source_file, device_id, "
            "timestamp, sequence_index, raw_attack, main_probabilities, "
            "tiny_benign_probability, boolean_fast_path FROM prediction_rows "
            f"ORDER BY {order}"
        )
        while True:
            chronological_rows = cast(
                list[tuple[Any, ...]],
                chronological.fetchmany(max(dataset.batch_size, 2)),
            )
            if not chronological_rows:
                break
            main_probability = np.vstack(
                [
                    np.frombuffer(row[8], dtype="<f4", count=len(CANONICAL_LABELS))
                    for row in chronological_rows
                ]
            )
            metadata = pd.DataFrame(
                {
                    "storage_position": [row[0] for row in chronological_rows],
                    "row_uid": [row[1] for row in chronological_rows],
                    "source_file": [row[3] for row in chronological_rows],
                    "device_id": [row[4] for row in chronological_rows],
                    "timestamp": [
                        np.nan if row[5] is None else row[5]
                        for row in chronological_rows
                    ],
                    "sequence_index": [row[6] for row in chronological_rows],
                }
            )
            routed, stages, _scores = router.route_batch(
                metadata,
                np.asarray([row[9] for row in chronological_rows], dtype=np.float32),
                main_probability,
                np.asarray([row[10] for row in chronological_rows], dtype=bool),
            )
            connection.executemany(
                "UPDATE prediction_rows SET routed_probabilities = ?, exit_stage = ? "
                "WHERE storage_position = ?",
                [
                    (
                        sqlite3.Binary(
                            np.asarray(routed[index], dtype="<f4").tobytes()
                        ),
                        int(stages[index]),
                        int(row[0]),
                    )
                    for index, row in enumerate(chronological_rows)
                ],
            )
            connection.commit()

        restored = connection.execute(
            "SELECT storage_position, row_uid, true_label, source_file, device_id, "
            "timestamp, sequence_index, raw_attack, routed_probabilities, exit_stage "
            "FROM prediction_rows ORDER BY storage_position"
        )
        restored_rows = 0
        while True:
            restored_batch_rows = cast(
                list[tuple[Any, ...]],
                restored.fetchmany(max(dataset.batch_size, 2)),
            )
            if not restored_batch_rows:
                break
            if any(row[8] is None or row[9] is None for row in restored_batch_rows):
                raise RuntimeError("temporal prediction routing omitted stored rows")
            probabilities = np.vstack(
                [
                    np.frombuffer(row[8], dtype="<f4", count=len(CANONICAL_LABELS))
                    for row in restored_batch_rows
                ]
            )
            stages = np.asarray([row[9] for row in restored_batch_rows], dtype=np.int8)
            restored_rows += len(restored_batch_rows)
            yield output_batch(restored_batch_rows, probabilities, stages)
        if restored_rows != seen:
            raise RuntimeError("temporal prediction restore lost row coverage")
    finally:
        if connection is not None:
            connection.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            database_path.with_name(database_path.name + suffix).unlink(missing_ok=True)


def run_out_of_core_training(
    config_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    prepared_descriptor_path: str | Path | None = None,
) -> Path:
    """Run exact neural training over immutable Parquet shards.

    Model compatibility is checked before constructing a dataset or run directory.
    The complete prepared-data contract is then verified before any run artifact is
    allocated.
    """

    path = Path(config_path)
    loaded = load_config(path) if config is None else config
    if str(loaded["dataset"].get("storage", "csv")) != "parquet":
        raise ValueError("out-of-core training requires dataset.storage=parquet")
    _reject_unsupported_model(loaded)
    prepared = (
        verify_prepared_dataset(prepared_descriptor_path)
        if prepared_descriptor_path is not None
        else _resolve_prepared_descriptor(path, loaded)
    )

    # Imported lazily to keep the storage dispatch boundary acyclic.
    return _run_verified_neural_training(loaded, prepared)


def _run_verified_neural_training(
    config: dict[str, Any], prepared: PreparedDataset
) -> Path:
    """Execute bounded train, calibration, evaluation, and export phases."""

    from dataclasses import replace

    import torch

    from bitguard_bnn.models import feature_gate_summary, parameter_summary
    from bitguard_bnn.out_of_core.cache import CacheLayout
    from bitguard_bnn.out_of_core.calibrate import (
        calibrate_fixed_fpr_from_cache,
        calibrate_open_set_from_cache,
        populate_validation_cache,
        route_validation_cache,
        tune_exit_threshold_from_cache,
        validation_contract,
    )
    from bitguard_bnn.out_of_core.trainer import fit_neural_streaming
    from bitguard_bnn.trainer import (
        _build_neural,
        _checkpoint,
        _process_resource_summary,
    )

    verified_config = load_config(prepared.resolved_config_path)
    if str(verified_config["dataset"].get("storage", "csv")) != "parquet":
        raise ValueError("verified prepared config requires dataset.storage=parquet")
    _reject_unsupported_model(verified_config)
    _require_matching_scientific_contract(config, verified_config)
    runtime_config = copy.deepcopy(verified_config)
    runtime_config["experiment"]["output_dir"] = config["experiment"]["output_dir"]
    seed = int(runtime_config["experiment"]["seed"])
    seed_everything(seed)

    # The caller verifies the complete prepared descriptor before reaching this
    # first allocation. Never move create_run_dir above that boundary.
    run_dir = create_run_dir(runtime_config)
    temporary = run_dir / "temporary"
    temporary.mkdir(parents=True, exist_ok=False)
    phases: list[dict[str, Any]] = []
    save_yaml(runtime_config, run_dir / "resolved_config.yaml")
    save_json(environment_manifest(), run_dir / "environment.json")
    save_json(prepared.to_dict(), run_dir / "prepared_dataset.json")
    shutil.copy2(prepared.preprocessor_path, run_dir / "preprocessor.joblib")
    preprocessor = FeaturePreprocessor.load(run_dir / "preprocessor.joblib")
    save_json(preprocessor.feature_manifest(), run_dir / "feature_manifest.json")

    labels = list(preprocessor.active_labels)
    counts = _class_counts(prepared, labels)
    attack_prior = _attack_prior(counts)
    training_contract = {
        "algorithm": "bitguard.full-validation-selection.v1",
        "prepared_descriptor_fingerprint": prepared.to_dict()["fingerprint"],
        "split_fingerprint": prepared.split_fingerprint,
        "validation_rows": prepared.validation_count,
    }
    batch_size = int(runtime_config["training"]["batch_size"])
    validation_dataset = _dataset(prepared, runtime_config, "validation")
    validation_callback = _validation_callback(
        validation_dataset, labels, temporary / "epoch-validation"
    )

    teacher_model: Any | None = None
    if float(runtime_config["loss"].get("distillation_alpha", 0.0)) > 0.0:
        with _phase(phases, "teacher_training", temporary):
            teacher_config = copy.deepcopy(runtime_config)
            teacher_config["model"]["type"] = "fp32_mlp"
            teacher_config["loss"]["distillation_alpha"] = 0.0
            teacher_config["loss"]["lambda_feature"] = 0.0
            teacher_model = _build_neural(teacher_config, preprocessor, len(labels))
            teacher_fit = fit_neural_streaming(
                teacher_model,
                _dataset(prepared, runtime_config, "train"),
                counts,
                labels,
                teacher_config,
                validation_callback,
                {**training_contract, "role": "teacher"},
                checkpoint_path=run_dir / "teacher_training_state.pt",
                progress_path=run_dir / "teacher_training_history.partial.csv",
                training_role="teacher",
            )
            teacher_model = teacher_fit.model
            teacher_fit.history.to_csv(
                run_dir / "teacher_training_history.csv", index=False
            )
            _checkpoint(
                teacher_model,
                teacher_config,
                preprocessor,
                run_dir / "teacher_model.pt",
                labels,
                list(runtime_config["model"]["hidden_dims"]),
            )

    with _phase(phases, "main_training", temporary):
        main_model = _build_neural(runtime_config, preprocessor, len(labels))
        resume_value = runtime_config["training"].get("resume_from")
        resume_path = (
            resolve_path(runtime_config, resume_value) if resume_value else None
        )
        main_fit = fit_neural_streaming(
            main_model,
            _dataset(prepared, runtime_config, "train"),
            counts,
            labels,
            runtime_config,
            validation_callback,
            {**training_contract, "role": "main"},
            teacher_model,
            checkpoint_path=run_dir / "last_training_state.pt",
            progress_path=run_dir / "training_history.partial.csv",
            resume_from=resume_path,
            training_role="main",
        )
        main_model = main_fit.model
        main_fit.history.to_csv(run_dir / "training_history.csv", index=False)
        _checkpoint(
            main_model,
            runtime_config,
            preprocessor,
            run_dir / "best_model.pt",
            labels,
            list(runtime_config["model"]["hidden_dims"]),
        )
        model_summary = {
            "model_type": runtime_config["model"]["type"],
            "artifact": "best_model.pt",
            "classes": labels,
            "best_validation_selection_score": main_fit.best_validation_score,
            "best_epoch": main_fit.best_epoch,
            "train_rows_per_epoch": prepared.train_count,
            "validation_rows_per_epoch": prepared.validation_count,
            **parameter_summary(main_model),
            **feature_gate_summary(main_model),
            **_process_resource_summary(run_dir / "best_model.pt"),
        }
        save_json(model_summary, run_dir / "model_summary.json")
    del teacher_model
    gc.collect()

    tiny_model: Any | None = None
    tiny_indices: np.ndarray | None = None
    if bool(runtime_config["cascade"].get("enabled", False)):
        with _phase(phases, "tiny_training", temporary):
            tiny_budget = min(
                int(runtime_config["cascade"]["tiny_feature_budget"]),
                len(preprocessor.selected_features),
            )
            tiny_indices = preprocessor.encoder.encoded_indices_for_first(
                tiny_budget, len(preprocessor.selected_features)
            )
            tiny_config = copy.deepcopy(runtime_config)
            tiny_config["model"]["type"] = "vanilla_bnn"
            tiny_config["loss"]["distillation_alpha"] = 0.0
            tiny_model = _build_neural(
                tiny_config,
                preprocessor,
                2,
                input_indices=tiny_indices,
                hidden_dims=list(runtime_config["cascade"]["hidden_dims"]),
                force_bnn=True,
            )
            tiny_fit = fit_neural_streaming(
                tiny_model,
                _dataset(prepared, runtime_config, "train"),
                counts,
                labels,
                tiny_config,
                _validation_callback(
                    validation_dataset,
                    labels,
                    temporary / "tiny-epoch-validation",
                    feature_indices=tiny_indices,
                    binary_attack_target=True,
                ),
                {**training_contract, "role": "tiny"},
                checkpoint_path=run_dir / "tiny_training_state.pt",
                progress_path=run_dir / "tiny_training_history.partial.csv",
                training_role="tiny",
                feature_indices=tiny_indices.tolist(),
                binary_attack_target=True,
            )
            tiny_model = tiny_fit.model
            tiny_fit.history.to_csv(run_dir / "tiny_training_history.csv", index=False)
            _checkpoint(
                tiny_model,
                tiny_config,
                preprocessor,
                run_dir / "tiny_model.pt",
                ["benign", "attack"],
                list(runtime_config["cascade"]["hidden_dims"]),
                input_indices=tiny_indices,
            )
            save_json(
                {
                    "model_type": "vanilla_bnn",
                    "artifact": "tiny_model.pt",
                    "classes": ["benign", "attack"],
                    "best_validation_selection_score": tiny_fit.best_validation_score,
                    "best_epoch": tiny_fit.best_epoch,
                    "train_rows_per_epoch": prepared.train_count,
                    **parameter_summary(tiny_model),
                    **_process_resource_summary(run_dir / "tiny_model.pt"),
                },
                run_dir / "tiny_model_summary.json",
            )

    calibration_config = copy.deepcopy(runtime_config)
    boolean_calibration: Any | None = None
    calibration: CascadeCalibration | None = None
    fixed_fpr: dict[float, float] = {}
    cascade_active = tiny_model is not None and tiny_indices is not None
    calibration_model = tiny_model
    calibration_indices = tiny_indices
    if not cascade_active:
        calibration_config["cascade"]["boolean_fast_path_enabled"] = False
        calibration_config["temporal"]["enabled"] = False

        class MainOnlyTiny(torch.nn.Module):
            def forward(self, values: Any) -> Any:
                return torch.zeros(
                    (len(values), 2), dtype=values.dtype, device=values.device
                )

        calibration_model = MainOnlyTiny()
        calibration_indices = np.asarray([0], dtype=np.int64)

    assert calibration_model is not None and calibration_indices is not None
    with _phase(phases, "validation_calibration", temporary):
        device_width, source_width = _text_widths(validation_dataset)
        main_fingerprint = _sha256(run_dir / "best_model.pt")
        tiny_fingerprint = (
            _sha256(run_dir / "tiny_model.pt") if cascade_active else None
        )
        provisional = CacheLayout(
            prepared_descriptor_fingerprint=str(prepared.to_dict()["fingerprint"]),
            shard_fingerprint=prepared.shard_fingerprint,
            preprocessor_fingerprint=prepared.preprocessing_fingerprint,
            source_fingerprint=prepared.normalized_source_fingerprint,
            main_checkpoint_fingerprint=main_fingerprint,
            tiny_checkpoint_fingerprint=tiny_fingerprint,
            inference_contract_fingerprint="pending",
            split="validation",
            row_count=prepared.validation_count,
            main_class_labels=tuple(labels),
            routed_class_labels=tuple(CANONICAL_LABELS),
            true_class_labels=tuple(CANONICAL_LABELS),
            selected_features=tuple(preprocessor.selected_features),
            boolean_features=tuple(validation_dataset.boolean_features),
            device_id_width=device_width,
            source_id_width=source_width,
        )
        validation_inference_contract = validation_contract(
            {
                "prepared": prepared.to_dict()["fingerprint"],
                "main_checkpoint": main_fingerprint,
                "tiny_checkpoint": tiny_fingerprint or "disabled-main-only",
            },
            cache_base_fingerprint=provisional.inference_base_fingerprint,
            config=calibration_config,
            tiny_indices=calibration_indices,
            encoded_input_dimension=preprocessor.encoded_dimension,
        )
        layout = replace(
            provisional,
            inference_contract_fingerprint=validation_inference_contract,
        )
        cache_root = temporary / "validation-cache"
        with _temporary_calibration_cache(cache_root, layout) as cache:
            boolean_calibration = populate_validation_cache(
                cache,
                lambda: _iter_evaluation_batches(validation_dataset),
                main_model,
                calibration_model,
                calibration_indices,
                inference_contract_fingerprint=validation_inference_contract,
                boolean_fast_path_enabled=bool(
                    calibration_config["cascade"].get("boolean_fast_path_enabled", True)
                ),
                min_attack_recall=float(
                    calibration_config["cascade"]["min_attack_recall"]
                ),
                work_dir=temporary / "validation-populate",
                encoded_input_dimension=preprocessor.encoded_dimension,
                chunk_rows=max(batch_size, 2),
            )
            calibrate_open_set_from_cache(
                cache,
                preprocessor,
                temporary / "open-set",
                chunk_rows=max(batch_size, 2),
            )
            if cascade_active:
                calibration = tune_exit_threshold_from_cache(
                    cache,
                    float(calibration_config["cascade"]["min_attack_recall"]),
                    int(calibration_config["cascade"]["threshold_grid_size"]),
                    float(calibration_config["cascade"]["false_negative_cost"]),
                    temporary / "exit-threshold",
                    chunk_rows=max(batch_size, 2),
                )
            else:
                calibration = CascadeCalibration(
                    exit_threshold=2.0,
                    attack_escalation_recall=1.0,
                    benign_early_exit_ratio=0.0,
                    overall_early_exit_ratio=0.0,
                    validation_rows=prepared.validation_count,
                    false_negative_cost=float(
                        calibration_config["cascade"]["false_negative_cost"]
                    ),
                )
            route_validation_cache(
                cache,
                preprocessor,
                calibration,
                calibration_config,
                attack_prior,
                validation_contract=validation_inference_contract,
                work_dir=temporary / "validation-routing",
                chunk_rows=max(batch_size, 2),
            )
            fixed_fpr = calibrate_fixed_fpr_from_cache(
                cache,
                runtime_config["evaluation"].get("fixed_fpr_targets", []),
                temporary / "fixed-fpr",
                chunk_rows=max(batch_size, 2),
            )
        save_json(boolean_calibration.to_dict(), run_dir / "boolean_fast_path.json")
        if cascade_active:
            save_json(calibration.to_dict(), run_dir / "cascade_calibration.json")
        save_json(
            {
                "threshold_source": "validation",
                "score_pipeline": "routed_attack_score=1-p_benign",
                "thresholds": {str(key): value for key, value in fixed_fpr.items()},
            },
            run_dir / "fixed_fpr_thresholds.json",
        )
        preprocessor.save(run_dir / "preprocessor.joblib")
        save_json(preprocessor.feature_manifest(), run_dir / "feature_manifest.json")
        save_yaml(preprocessor.config, run_dir / "calibrated_config.yaml")

    del validation_dataset
    gc.collect()
    if calibration is None or boolean_calibration is None:
        raise RuntimeError(
            "validation calibration did not produce an inference contract"
        )
    prediction_metadata = _prediction_temporal_contract(prepared, runtime_config)
    inference_contract = _test_inference_contract(
        prepared=prepared,
        run_dir=run_dir,
        preprocessor=preprocessor,
        runtime_config=runtime_config,
        tiny_indices=tiny_indices,
        calibration=calibration,
        boolean_calibration=boolean_calibration,
        fixed_fpr=fixed_fpr,
        attack_prior=attack_prior,
        prediction_metadata=prediction_metadata,
    )
    save_json(inference_contract, run_dir / "inference_contract.json")
    with _phase(phases, "test_evaluation", temporary):
        test_dataset = _dataset(prepared, runtime_config, "test")
        evaluation = evaluate_prediction_batches(
            lambda: _prediction_batches(
                test_dataset,
                main_model,
                preprocessor,
                tiny_model=tiny_model,
                tiny_indices=tiny_indices,
                calibration=calibration,
                boolean_calibration=boolean_calibration,
                config=runtime_config,
                attack_prior=attack_prior,
                prediction_metadata=prediction_metadata,
                temporary_directory=temporary / "test-routing",
            ),
            probability_labels=CANONICAL_LABELS,
            high_risk_labels=runtime_config["evaluation"]["high_risk_labels"],
            test_contract=str(inference_contract["fingerprint"]),
            prediction_path=run_dir / "predictions.parquet",
            metrics_path=run_dir / "metrics.json",
            plot_sample_path=run_dir / "plot_sample.parquet",
            plot_manifest_path=run_dir / "plot_manifest.json",
            temporary_directory=temporary / "evaluation",
            operating_thresholds=fixed_fpr,
            plot_sample_rows=int(
                runtime_config["evaluation"].get("plot_sample_rows", 50_000)
            ),
            plot_sample_seed=seed,
            score_run_rows=max(batch_size, 2),
        )
        if int(evaluation["rows"]) != prepared.test_count:
            raise RuntimeError(
                "test evaluation row count does not match prepared split"
            )

    with _phase(phases, "edge_export", temporary):
        export_result = export_run(run_dir, run_dir / "edge")

    save_json(phases, run_dir / "phase_resources.json")
    save_json(
        {
            "dataset": prepared.dataset,
            "prepared_descriptor": prepared.descriptor_path,
            "train_rows_per_epoch": prepared.train_count,
            "validation_rows": prepared.validation_count,
            "test_rows": prepared.test_count,
            "resume_checkpoint": str(run_dir / "last_training_state.pt"),
            "best_checkpoint": str(run_dir / "best_model.pt"),
            "metrics": str(run_dir / "metrics.json"),
            "predictions": str(run_dir / "predictions.parquet"),
            "inference_contract": str(run_dir / "inference_contract.json"),
            "export": export_result,
        },
        run_dir / "run_summary.json",
    )
    shutil.rmtree(temporary)
    return run_dir


__all__ = ["run_out_of_core_training"]
