"""Exact, restartable calibration over the disk-backed validation cache."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cascade import (
    BooleanFastPathCalibration,
    CascadeCalibration,
    CascadeStreamRouter,
)
from ..preprocess import FeaturePreprocessor
from .cache import CalibrationCache


CALIBRATION_ALGORITHM = "bitguard.streaming-calibration.v1"
ORDER_ALGORITHM = "bitguard.validation-order.timestamp-device-uid-position.v1"
_MAX_MERGE_FAN_IN = 32
_BOOLEAN_ALGORITHM = "bitguard.boolean-fast-path.exact-quantile.v1"
_TINY_INPUT_ALGORITHM = "bitguard.tiny-input-indices.v1"


def _normalized_tiny_indices(
    values: Sequence[int] | np.ndarray[Any, Any], input_dimension: int
) -> np.ndarray[Any, Any]:
    if type(input_dimension) is not int or input_dimension <= 0:
        raise ValueError("encoded_input_dimension must be a positive integer")
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("tiny_indices must be a non-empty integer vector")
    indices = raw.astype(np.int64, copy=True)
    if (
        np.any(indices < 0)
        or np.any(indices >= input_dimension)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("tiny_indices must be unique and inside the encoded input")
    return indices


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validation_contract(
    identity: Mapping[str, object],
    *,
    cache_base_fingerprint: str,
    config: Mapping[str, Any],
    tiny_indices: Sequence[int] | np.ndarray[Any, Any],
    encoded_input_dimension: int,
) -> str:
    """Bind validation inference to its immutable data and open-set settings."""

    if not cache_base_fingerprint:
        raise ValueError("cache_base_fingerprint must be non-empty")
    boolean_policy = {
        "algorithm": _BOOLEAN_ALGORITHM,
        "enabled": bool(config["cascade"].get("boolean_fast_path_enabled", True)),
        "min_attack_recall": float(config["cascade"]["min_attack_recall"]),
    }
    normalized_indices = _normalized_tiny_indices(
        tiny_indices, encoded_input_dimension
    )
    tiny_policy = {
        "algorithm": _TINY_INPUT_ALGORITHM,
        "encoded_input_dimension": encoded_input_dimension,
        "indices": normalized_indices.tolist(),
    }
    contract = _fingerprint(
        {
            "algorithm": CALIBRATION_ALGORITHM,
            "identity": dict(identity),
            "cache_base_fingerprint": cache_base_fingerprint,
            "open_set": dict(config["preprocess"]["open_set"]),
            "boolean_policy": boolean_policy,
            "tiny_input": tiny_policy,
        }
    )
    return ".".join(
        (
            cache_base_fingerprint,
            contract,
            _fingerprint(boolean_policy),
            _fingerprint(tiny_policy),
        )
    )


def _validate_inference_contract(
    cache: CalibrationCache,
    contract: str,
    *,
    enabled: bool,
    min_attack_recall: float,
    tiny_indices: np.ndarray[Any, Any],
    encoded_input_dimension: int,
) -> None:
    parts = contract.split(".")
    if len(parts) != 4 or parts[0] != cache.layout.inference_base_fingerprint:
        raise ValueError("validation inference contract cache base mismatch")
    policy_fingerprint = parts[2]
    expected = _fingerprint(
        {
            "algorithm": _BOOLEAN_ALGORITHM,
            "enabled": bool(enabled),
            "min_attack_recall": float(min_attack_recall),
        }
    )
    if policy_fingerprint != expected:
        raise ValueError("validation inference contract Boolean policy mismatch")
    tiny_policy = {
        "algorithm": _TINY_INPUT_ALGORITHM,
        "encoded_input_dimension": encoded_input_dimension,
        "indices": tiny_indices.tolist(),
    }
    if parts[3] != _fingerprint(tiny_policy):
        raise ValueError("validation inference contract Tiny input mismatch")


def populate_validation_cache(
    cache: CalibrationCache,
    batch_factory: Callable[[], Iterable[Mapping[str, Any]]],
    main_model: Any,
    tiny_model: Any,
    tiny_indices: np.ndarray,
    *,
    inference_contract_fingerprint: str,
    boolean_fast_path_enabled: bool,
    min_attack_recall: float,
    work_dir: Path | str,
    encoded_input_dimension: int,
    device: Any | None = None,
    chunk_rows: int = 65_536,
    stop_after_inference_rows: int | None = None,
) -> BooleanFastPathCalibration:
    """Run bounded Main/Tiny validation inference and durably append cache rows.

    Boolean thresholds require the complete raw validation distribution, so an
    enabled Boolean path deliberately performs a raw-only first pass.  The
    second pass applies those frozen thresholds while running both models.
    """

    _require_validation_split(cache)
    indices = _normalized_tiny_indices(tiny_indices, encoded_input_dimension)
    if cache.readonly:
        raise ValueError("validation inference requires a writable cache")
    if inference_contract_fingerprint != cache.layout.inference_contract_fingerprint:
        raise ValueError("validation inference contract fingerprint mismatch")
    _validate_inference_contract(
        cache,
        inference_contract_fingerprint,
        enabled=boolean_fast_path_enabled,
        min_attack_recall=min_attack_recall,
        tiny_indices=indices,
        encoded_input_dimension=encoded_input_dimension,
    )
    if cache.routed_committed_rows:
        raise ValueError("validation inference cannot resume after routing has begun")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if stop_after_inference_rows is not None and (
        type(stop_after_inference_rows) is not int
        or not 0 <= stop_after_inference_rows <= cache.layout.row_count
        or stop_after_inference_rows < cache.committed_rows
    ):
        raise ValueError("stop_after_inference_rows is outside the cache layout")
    boolean_features = cache.layout.boolean_features
    if boolean_fast_path_enabled and boolean_features:
        def raw_batches() -> Iterator[tuple[np.ndarray, Mapping[str, np.ndarray]]]:
            for batch in batch_factory():
                metadata = batch.get("metadata")
                raw = batch.get("boolean_raw")
                if not isinstance(metadata, Mapping) or not isinstance(raw, Mapping):
                    raise ValueError("validation batch metadata/boolean_raw is invalid")
                yield np.asarray(metadata["behavior_label"], dtype=str), {
                    name: np.asarray(raw[name]) for name in boolean_features
                }

        boolean_calibration = tune_boolean_fast_path_streaming(
            raw_batches(),
            row_count=cache.layout.row_count,
            features=boolean_features,
            min_attack_recall=min_attack_recall,
            work_dir=Path(work_dir) / "boolean",
            chunk_rows=chunk_rows,
        )
    else:
        boolean_calibration = BooleanFastPathCalibration(
            False, list(boolean_features), {}, 1.0, 0.0
        )

    import torch

    if device is None:
        device = next(main_model.parameters()).device
    previous_main_training = bool(main_model.training)
    previous_tiny_training = bool(tiny_model.training)
    main_model.eval()
    tiny_model.eval()
    seen = 0
    committed = cache.committed_rows
    try:
        with torch.inference_mode():
            for batch in batch_factory():
                features = np.asarray(batch["features"], dtype=np.float32)
                unencoded = np.asarray(batch["unencoded"], dtype=np.float32)
                row_uids = np.asarray(batch["row_uid"], dtype=str)
                metadata = batch.get("metadata")
                raw = batch.get("boolean_raw")
                if not isinstance(metadata, Mapping) or not isinstance(raw, Mapping):
                    raise ValueError("validation batch metadata/boolean_raw is invalid")
                rows = len(row_uids)
                if (
                    rows <= 0
                    or features.ndim != 2
                    or features.shape[0] != rows
                    or features.shape[1] != encoded_input_dimension
                    or unencoded.shape != (rows, len(cache.layout.selected_features))
                    or seen + rows > cache.layout.row_count
                ):
                    raise ValueError("validation batch has inconsistent row coverage")
                batch_start = seen
                batch_end = seen + rows
                seen = batch_end
                if batch_end <= committed:
                    expected = np.vstack(
                        [
                            np.frombuffer(bytes.fromhex(uid), dtype=np.uint8)
                            for uid in row_uids
                        ]
                    )
                    if not np.array_equal(
                        cache.arrays["uid_digest"][batch_start:batch_end], expected
                    ):
                        raise RuntimeError(
                            "validation batch order differs from the committed cache prefix"
                        )
                    continue
                local_start = max(0, committed - batch_start)
                local_end = rows
                if local_start:
                    expected_prefix = np.vstack(
                        [
                            np.frombuffer(bytes.fromhex(uid), dtype=np.uint8)
                            for uid in row_uids[:local_start]
                        ]
                    )
                    if not np.array_equal(
                        cache.arrays["uid_digest"][
                            batch_start : batch_start + local_start
                        ],
                        expected_prefix,
                    ):
                        raise RuntimeError(
                            "validation batch order differs from the committed cache prefix"
                        )
                if stop_after_inference_rows is not None:
                    local_end = min(local_end, stop_after_inference_rows - batch_start)
                if local_end <= local_start:
                    break
                main_tensor = torch.from_numpy(features).to(device)
                tiny_tensor = torch.from_numpy(features[:, indices]).to(device)
                known_full = torch.softmax(main_model(main_tensor), dim=1).cpu().numpy().astype(
                    np.float32, copy=False
                )
                tiny_full = (
                    torch.softmax(tiny_model(tiny_tensor), dim=1)[:, 0]
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                known = known_full[local_start:local_end]
                tiny = tiny_full[local_start:local_end]
                actual_rows = local_end - local_start
                if known_full.shape != (
                    rows,
                    len(cache.layout.main_class_labels),
                ) or tiny_full.shape != (rows,):
                    raise ValueError("validation model output shape does not match cache layout")
                uid_values: list[np.ndarray] = []
                for uid in row_uids[local_start:local_end]:
                    try:
                        encoded_uid = bytes.fromhex(str(uid))
                    except ValueError as error:
                        raise ValueError("row_uid must be a 64-character hex digest") from error
                    if len(encoded_uid) != 32:
                        raise ValueError("row_uid must be a 64-character hex digest")
                    uid_values.append(np.frombuffer(encoded_uid, dtype=np.uint8))
                true_names = np.asarray(
                    metadata["behavior_label"], dtype=str
                )[local_start:local_end]
                true_lookup = {
                    label: index for index, label in enumerate(cache.layout.true_class_labels)
                }
                try:
                    true_labels = np.asarray(
                        [true_lookup[name] for name in true_names], dtype=np.int32
                    )
                except KeyError as error:
                    raise ValueError("validation label is absent from cache layout") from error
                flags: np.ndarray[Any, Any] = np.zeros(
                    (actual_rows, len(boolean_features)), dtype=np.bool_
                )
                if boolean_calibration.enabled:
                    for column, name in enumerate(boolean_features):
                        values = np.asarray(raw[name], dtype=np.float64)[local_start:local_end]
                        threshold = boolean_calibration.upper_thresholds[name]
                        flags[:, column] = np.isfinite(values) & (values <= threshold)
                cache.commit_inference_range(
                    cache.committed_rows,
                    {
                        "cache_position": np.arange(
                            batch_start + local_start,
                            batch_start + local_end,
                            dtype=np.int64,
                        ),
                        "uid_digest": np.vstack(uid_values),
                        "true_label": true_labels,
                        "known_probabilities": known,
                        "selected_values": unencoded[local_start:local_end],
                        "tiny_benign_probability": tiny,
                        "timestamp": np.asarray(
                            metadata["timestamp"], dtype=np.float64
                        )[local_start:local_end],
                        "sequence": np.asarray(
                            metadata["sequence_index"], dtype=np.int64
                        )[local_start:local_end],
                        "device_id": np.asarray(
                            metadata["device_id"], dtype=str
                        )[local_start:local_end].tolist(),
                        "source_id": np.asarray(
                            metadata["source_file"], dtype=str
                        )[local_start:local_end].tolist(),
                        "boolean_flags": flags,
                    },
                )
                committed = cache.committed_rows
                if (
                    stop_after_inference_rows is not None
                    and committed >= stop_after_inference_rows
                ):
                    break
        if stop_after_inference_rows is None and seen != cache.layout.row_count:
            raise ValueError("validation batches do not fill the cache layout")
        return boolean_calibration
    finally:
        main_model.train(previous_main_training)
        tiny_model.train(previous_tiny_training)


def _require_validation_split(cache: CalibrationCache) -> None:
    if cache.layout.split != "validation":
        raise ValueError("calibration is restricted to the validation split")


def _require_complete_inference(cache: CalibrationCache) -> None:
    _require_validation_split(cache)
    if cache.committed_rows != cache.layout.row_count:
        raise ValueError("calibration requires a complete validation inference cache")


def _close_memmap(array: np.memmap[Any, Any] | None) -> None:
    if array is None:
        return
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _temporary_memmap(
    path: Path, dtype: np.dtype[Any] | str, shape: tuple[int, ...]
) -> np.memmap[Any, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def calibrate_open_set_from_cache(
    cache: CalibrationCache,
    preprocessor: FeaturePreprocessor,
    work_dir: Path | str,
    *,
    chunk_rows: int = 65_536,
) -> tuple[float, float]:
    """Match ``FeaturePreprocessor`` calibration without materializing validation."""

    _require_complete_inference(cache)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if tuple(preprocessor.active_labels) != cache.layout.main_class_labels:
        raise ValueError("preprocessor labels do not match the calibration cache")
    if tuple(preprocessor.selected_features) != cache.layout.selected_features:
        raise ValueError("preprocessor features do not match the calibration cache")
    root = Path(work_dir)
    benign_index = cache.layout.true_class_labels.index("benign")
    labels = cache.arrays["true_label"]
    benign_count = 0
    for start in range(0, cache.layout.row_count, chunk_rows):
        end = min(start + chunk_rows, cache.layout.row_count)
        benign_count += int(np.count_nonzero(labels[start:end] == benign_index))

    distance_path = root / "benign-distance.f32"
    distance_values: np.memmap[Any, Any] | None = None
    try:
        if benign_count:
            distance_values = _temporary_memmap(
                distance_path, np.dtype("<f4"), (benign_count,)
            )
            offset = 0
            for start in range(0, cache.layout.row_count, chunk_rows):
                end = min(start + chunk_rows, cache.layout.row_count)
                mask = np.asarray(labels[start:end]) == benign_index
                if not mask.any():
                    continue
                distances = preprocessor.anomaly_distance(
                    np.asarray(cache.arrays["selected_values"][start:end])[mask]
                )
                distance_values[offset : offset + len(distances)] = distances
                offset += len(distances)
            quantile = float(
                preprocessor.config["preprocess"]["open_set"].get(
                    "benign_distance_quantile", 0.99
                )
            )
            preprocessor.open_distance_threshold = max(
                float(np.quantile(distance_values, quantile, overwrite_input=True)),
                1e-6,
            )
    finally:
        _close_memmap(distance_values)
        distance_path.unlink(missing_ok=True)

    anomaly_count = 0
    for start in range(0, cache.layout.row_count, chunk_rows):
        end = min(start + chunk_rows, cache.layout.row_count)
        distances = preprocessor.anomaly_distance(
            np.asarray(cache.arrays["selected_values"][start:end])
        )
        anomaly_count += int(
            np.count_nonzero(distances >= preprocessor.open_distance_threshold)
        )
    maximum_path = root / "anomaly-maximum.f32"
    maximum_values: np.memmap[Any, Any] | None = None
    try:
        if anomaly_count:
            maximum_values = _temporary_memmap(
                maximum_path, np.dtype("<f4"), (anomaly_count,)
            )
            offset = 0
            for start in range(0, cache.layout.row_count, chunk_rows):
                end = min(start + chunk_rows, cache.layout.row_count)
                distances = preprocessor.anomaly_distance(
                    np.asarray(cache.arrays["selected_values"][start:end])
                )
                mask = distances >= preprocessor.open_distance_threshold
                values = np.asarray(cache.arrays["known_probabilities"][start:end]).max(
                    axis=1
                )[mask]
                maximum_values[offset : offset + len(values)] = values
                offset += len(values)
            maximum_values.sort()
        target = float(
            preprocessor.config["preprocess"]["open_set"].get(
                "max_known_false_unknown_rate", 0.02
            )
        )
        if not 0.0 <= target < 0.5:
            raise ValueError("max_known_false_unknown_rate must be in [0, 0.5)")
        chosen = 0.0
        for threshold in np.linspace(0.05, 0.99, 189):
            false_unknown = (
                0
                if maximum_values is None
                else int(np.searchsorted(maximum_values, threshold, side="left"))
            )
            rate = false_unknown / cache.layout.row_count
            if rate <= target + 1e-12:
                chosen = float(threshold)
            else:
                break
        preprocessor.config["preprocess"]["open_set"]["confidence_threshold"] = chosen
    finally:
        _close_memmap(maximum_values)
        maximum_path.unlink(missing_ok=True)
    return float(preprocessor.open_distance_threshold), chosen


def tune_boolean_fast_path_streaming(
    batches: Iterable[tuple[np.ndarray, Mapping[str, np.ndarray]]],
    *,
    row_count: int,
    features: Sequence[str],
    min_attack_recall: float,
    work_dir: Path | str,
    chunk_rows: int = 65_536,
) -> BooleanFastPathCalibration:
    """Tune Boolean thresholds exactly from disk-backed raw feature vectors."""

    if row_count <= 0 or chunk_rows <= 0:
        raise ValueError("row_count and chunk_rows must be positive")
    names = list(features)
    if len(set(names)) != len(names):
        raise ValueError("Boolean feature names must be unique")
    if not names:
        return BooleanFastPathCalibration(False, [], {}, 1.0, 0.0)
    root = Path(work_dir)
    label_path = root / "boolean-benign.u1"
    label_values: np.memmap[Any, Any] | None = None
    raw_values: dict[str, np.memmap[Any, Any]] = {}
    raw_paths = {name: root / f"boolean-{index}.f64" for index, name in enumerate(names)}
    try:
        label_values = _temporary_memmap(label_path, np.dtype("|u1"), (row_count,))
        raw_values = {
            name: _temporary_memmap(raw_paths[name], np.dtype("<f8"), (row_count,))
            for name in names
        }
        offset = 0
        for labels, values in batches:
            labels_array = np.asarray(labels, dtype=str)
            rows = len(labels_array)
            if rows <= 0 or offset + rows > row_count:
                raise ValueError("Boolean calibration batches exceed declared row_count")
            if set(values) != set(names):
                raise ValueError("Boolean calibration batch features do not match")
            label_values[offset : offset + rows] = labels_array == "benign"
            for name in names:
                feature_values = np.asarray(values[name], dtype=np.float64)
                if feature_values.shape != (rows,):
                    raise ValueError("Boolean calibration feature has an invalid shape")
                raw_values[name][offset : offset + rows] = feature_values
            offset += rows
        if offset != row_count:
            raise ValueError("Boolean calibration batches do not fill declared row_count")
        benign_count = int(np.count_nonzero(label_values))
        attack_count = row_count - benign_count
        if not benign_count or not attack_count:
            return BooleanFastPathCalibration(False, names, {}, 1.0, 0.0)

        quantiles = np.linspace(0.50, 0.99, 50)
        threshold_table: dict[str, np.ndarray] = {}
        for name in names:
            finite_benign_count = 0
            feature_memmap = raw_values[name]
            for start in range(0, row_count, chunk_rows):
                end = min(start + chunk_rows, row_count)
                block = np.asarray(feature_memmap[start:end])
                finite_benign_count += int(
                    np.count_nonzero(
                        np.asarray(label_values[start:end], dtype=bool) & ~np.isnan(block)
                    )
                )
            if not finite_benign_count:
                threshold_table[name] = np.full(len(quantiles), np.nan)
                continue
            compact_path = root / f"boolean-benign-{names.index(name)}.f64"
            compact: np.memmap[Any, Any] | None = None
            try:
                compact = _temporary_memmap(
                    compact_path, np.dtype("<f8"), (finite_benign_count,)
                )
                compact_offset = 0
                for start in range(0, row_count, chunk_rows):
                    end = min(start + chunk_rows, row_count)
                    block = np.asarray(feature_memmap[start:end])
                    mask = np.asarray(label_values[start:end], dtype=bool) & ~np.isnan(
                        block
                    )
                    selected = block[mask]
                    compact[compact_offset : compact_offset + len(selected)] = selected
                    compact_offset += len(selected)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    threshold_table[name] = np.asarray(
                        np.quantile(compact, quantiles, overwrite_input=True),
                        dtype=np.float64,
                    )
            finally:
                _close_memmap(compact)
                compact_path.unlink(missing_ok=True)

        best: tuple[float, dict[str, float], float] | None = None
        for quantile_index in range(len(quantiles)):
            thresholds = {
                name: float(threshold_table[name][quantile_index]) for name in names
            }
            benign_exit_count = 0
            attack_escalation_count = 0
            for start in range(0, row_count, chunk_rows):
                end = min(start + chunk_rows, row_count)
                mask = np.ones(end - start, dtype=bool)
                for name, threshold in thresholds.items():
                    block = np.asarray(raw_values[name][start:end])
                    mask &= np.isfinite(block) & (block <= threshold)
                benign = np.asarray(label_values[start:end], dtype=bool)
                benign_exit_count += int(np.count_nonzero(mask & benign))
                attack_escalation_count += int(np.count_nonzero(~mask & ~benign))
            attack_recall = attack_escalation_count / attack_count
            if attack_recall + 1e-12 < min_attack_recall:
                continue
            benign_exit = benign_exit_count / benign_count
            if best is None or benign_exit > best[0]:
                best = (benign_exit, thresholds, attack_recall)
        if best is None:
            return BooleanFastPathCalibration(False, names, {}, 1.0, 0.0)
        return BooleanFastPathCalibration(True, names, best[1], best[2], best[0])
    finally:
        for array in raw_values.values():
            _close_memmap(array)
        _close_memmap(label_values)
        for path in [label_path, *raw_paths.values()]:
            path.unlink(missing_ok=True)


def tune_exit_threshold_from_cache(
    cache: CalibrationCache,
    min_attack_recall: float,
    grid_size: int,
    false_negative_cost: float,
    work_dir: Path | str,
    *,
    chunk_rows: int = 65_536,
) -> CascadeCalibration:
    """Find the same monotone optimum as ``tune_exit_threshold`` on disk."""

    _require_complete_inference(cache)
    if grid_size <= 0 or chunk_rows <= 0:
        raise ValueError("grid_size and chunk_rows must be positive")
    benign_index = cache.layout.true_class_labels.index("benign")
    rows = cache.layout.row_count
    labels = cache.arrays["true_label"]
    attack_count = 0
    benign_count = 0
    score_min = math.inf
    score_max = -math.inf
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        block_labels = np.asarray(labels[start:end])
        benign_count += int(np.count_nonzero(block_labels == benign_index))
        attack_count += int(np.count_nonzero(block_labels != benign_index))
        scores = 2.0 * np.asarray(
            cache.arrays["tiny_benign_probability"][start:end], dtype=np.float64
        ) - 1.0
        score_min = min(score_min, float(scores.min()))
        score_max = max(score_max, float(scores.max()))
    if not benign_count or not attack_count:
        raise ValueError("cascade calibration requires benign and attack validation rows")

    required = max(0, int(math.ceil((float(min_attack_recall) - 1e-12) * attack_count)))
    attack_path = Path(work_dir) / "attack-exit-score.f64"
    attacks: np.memmap[Any, Any] | None = None
    try:
        attacks = _temporary_memmap(attack_path, np.dtype("<f8"), (attack_count,))
        offset = 0
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            mask = np.asarray(labels[start:end]) != benign_index
            scores = 2.0 * np.asarray(
                cache.arrays["tiny_benign_probability"][start:end], dtype=np.float64
            ) - 1.0
            selected = scores[mask]
            attacks[offset : offset + len(selected)] = selected
            offset += len(selected)
        boundary = -math.inf
        if required > attack_count:
            boundary = score_max
        elif required:
            attacks.partition(required - 1)
            boundary = float(attacks[required - 1])

        candidate = math.inf
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            scores = 2.0 * np.asarray(
                cache.arrays["tiny_benign_probability"][start:end], dtype=np.float64
            ) - 1.0
            eligible = scores[scores > boundary]
            if eligible.size:
                candidate = min(candidate, float(eligible.min()))
        grid = np.linspace(score_min - 1e-6, score_max + 1e-6, grid_size)
        eligible_grid = grid[grid > boundary]
        if eligible_grid.size:
            candidate = min(candidate, float(eligible_grid.min()))
        if not math.isfinite(candidate):
            candidate = score_max + 1e-6

        benign_exit_count = 0
        overall_exit_count = 0
        attack_escalation_count = 0
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            block_labels = np.asarray(labels[start:end])
            scores = 2.0 * np.asarray(
                cache.arrays["tiny_benign_probability"][start:end], dtype=np.float64
            ) - 1.0
            early = scores >= candidate
            benign = block_labels == benign_index
            benign_exit_count += int(np.count_nonzero(early & benign))
            overall_exit_count += int(np.count_nonzero(early))
            attack_escalation_count += int(np.count_nonzero(~early & ~benign))
        return CascadeCalibration(
            candidate,
            attack_escalation_count / attack_count,
            benign_exit_count / benign_count,
            overall_exit_count / rows,
            rows,
            false_negative_cost,
        )
    finally:
        _close_memmap(attacks)
        attack_path.unlink(missing_ok=True)


def _decode_text(cache: CalibrationCache, prefix: str, positions: np.ndarray) -> list[str]:
    byte_values = cache.arrays[f"{prefix}_bytes"]
    lengths = cache.arrays[f"{prefix}_length"]
    return [
        bytes(byte_values[int(position), : int(lengths[int(position)])]).decode("utf-8")
        for position in positions
    ]


def _metadata(cache: CalibrationCache, positions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": _decode_text(cache, "source_id", positions),
            "device_id": _decode_text(cache, "device_id", positions),
            "timestamp": np.asarray(cache.arrays["timestamp"][positions], dtype=np.float64),
            "sequence_index": np.asarray(cache.arrays["sequence"][positions], dtype=np.int64),
            "row_uid": [
                bytes(cache.arrays["uid_digest"][int(position)]).hex()
                for position in positions
            ],
        }
    )


def _external_order_runs(
    cache: CalibrationCache,
    work_dir: Path,
    chunk_rows: int,
    *,
    use_timestamp: bool,
) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    for stale in work_dir.glob("order-*.npy"):
        stale.unlink(missing_ok=True)
    paths: list[Path] = []
    for run_index, start in enumerate(range(0, cache.layout.row_count, chunk_rows)):
        end = min(start + chunk_rows, cache.layout.row_count)
        positions: np.ndarray[Any, Any] = np.arange(start, end, dtype=np.int64)
        metadata = _metadata(cache, positions)
        if use_timestamp:
            frame = metadata.assign(__position=positions)
            ordered = frame.sort_values(
                ["timestamp", "device_id", "row_uid", "__position"], kind="stable"
            )["__position"].to_numpy(dtype=np.int64)
        else:
            frame = metadata.assign(__position=positions)
            ordered = frame.sort_values(
                ["device_id", "sequence_index", "row_uid", "__position"], kind="stable"
            )["__position"].to_numpy(dtype=np.int64)
        path = work_dir / f"order-{run_index:08d}.npy"
        np.save(path, ordered, allow_pickle=False)
        paths.append(path)
    return paths


def _position_key(cache: CalibrationCache, position: int, use_timestamp: bool) -> tuple[Any, ...]:
    positions = np.asarray([position], dtype=np.int64)
    device = _decode_text(cache, "device_id", positions)[0]
    uid = bytes(cache.arrays["uid_digest"][position])
    if use_timestamp:
        return (float(cache.arrays["timestamp"][position]), device, uid, position)
    return (device, int(cache.arrays["sequence"][position]), uid, position)


def _merge_order_runs(
    cache: CalibrationCache,
    paths: Sequence[Path],
    *,
    use_timestamp: bool,
) -> Iterator[int]:
    arrays: list[np.memmap[Any, Any]] = []
    heap: list[tuple[tuple[Any, ...], int, int]] = []
    try:
        for run_index, path in enumerate(paths):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays.append(array)
            if len(array):
                position = int(array[0])
                heapq.heappush(
                    heap, (_position_key(cache, position, use_timestamp), run_index, 0)
                )
        while heap:
            _, run_index, offset = heapq.heappop(heap)
            position = int(arrays[run_index][offset])
            yield position
            next_offset = offset + 1
            if next_offset < len(arrays[run_index]):
                next_position = int(arrays[run_index][next_offset])
                heapq.heappush(
                    heap,
                    (
                        _position_key(cache, next_position, use_timestamp),
                        run_index,
                        next_offset,
                    ),
                )
    finally:
        for array in arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()


def _write_merged_order_run(
    cache: CalibrationCache,
    paths: Sequence[Path],
    output_path: Path,
    *,
    use_timestamp: bool,
    buffer_rows: int,
) -> None:
    total = 0
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        total += len(array)
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.int64, shape=(total,)
    )
    buffer: np.ndarray[Any, Any] = np.empty(
        min(buffer_rows, max(total, 1)), dtype=np.int64
    )
    offset = 0
    buffered = 0
    try:
        for position in _merge_order_runs(
            cache, paths, use_timestamp=use_timestamp
        ):
            buffer[buffered] = position
            buffered += 1
            if buffered == len(buffer):
                output[offset : offset + buffered] = buffer
                offset += buffered
                buffered = 0
        if buffered:
            output[offset : offset + buffered] = buffer[:buffered]
            offset += buffered
        if offset != total:
            raise RuntimeError("external validation order lost row coverage")
        output.flush()
    finally:
        mapping = getattr(output, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _compact_order_runs(
    cache: CalibrationCache,
    paths: Sequence[Path],
    work_dir: Path,
    *,
    use_timestamp: bool,
    buffer_rows: int,
) -> list[Path]:
    current = list(paths)
    generation = 0
    while len(current) > _MAX_MERGE_FAN_IN:
        following: list[Path] = []
        for group_index, start in enumerate(
            range(0, len(current), _MAX_MERGE_FAN_IN)
        ):
            group = current[start : start + _MAX_MERGE_FAN_IN]
            output = work_dir / f"order-merge-{generation:04d}-{group_index:08d}.npy"
            _write_merged_order_run(
                cache,
                group,
                output,
                use_timestamp=use_timestamp,
                buffer_rows=buffer_rows,
            )
            following.append(output)
            for path in group:
                path.unlink(missing_ok=True)
        current = following
        generation += 1
    return current


def _all_present_timestamps(cache: CalibrationCache, chunk_rows: int) -> bool:
    for start in range(0, cache.layout.row_count, chunk_rows):
        end = min(start + chunk_rows, cache.layout.row_count)
        if np.isnan(cache.arrays["timestamp"][start:end]).any():
            return False
    return True


def _routing_contract(
    cache: CalibrationCache,
    preprocessor: FeaturePreprocessor,
    calibration: CascadeCalibration,
    config: Mapping[str, Any],
    attack_prior: np.ndarray,
    inference_validation_contract: str,
) -> str:
    if not inference_validation_contract:
        raise ValueError("validation_contract must be non-empty")
    return _fingerprint(
        {
            "algorithm": CALIBRATION_ALGORITHM,
            "order_algorithm": ORDER_ALGORITHM,
            "validation_contract": inference_validation_contract,
            "cache_layout_fingerprint": cache.layout.fingerprint,
            "calibration": calibration.to_dict(),
            "active_labels": list(preprocessor.active_labels),
            "open_distance_threshold": float(preprocessor.open_distance_threshold),
            "confidence_threshold": float(
                preprocessor.config["preprocess"]["open_set"]["confidence_threshold"]
            ),
            "cascade": dict(config["cascade"]),
            "temporal": dict(config["temporal"]),
            "attack_prior": np.asarray(attack_prior, dtype=np.float64).tolist(),
        }
    )


def route_validation_cache(
    cache: CalibrationCache,
    preprocessor: FeaturePreprocessor,
    calibration: CascadeCalibration,
    config: Mapping[str, Any],
    attack_prior: np.ndarray,
    *,
    validation_contract: str,
    work_dir: Path | str,
    chunk_rows: int = 65_536,
    stop_after_routed_rows: int | None = None,
) -> str:
    """Route validation in deterministic global order and resume exact state."""

    _require_complete_inference(cache)
    if cache.readonly:
        raise ValueError("routing requires a writable calibration cache")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if stop_after_routed_rows is not None and (
        type(stop_after_routed_rows) is not int
        or not 0 <= stop_after_routed_rows <= cache.layout.row_count
        or stop_after_routed_rows < cache.routed_committed_rows
    ):
        raise ValueError("stop_after_routed_rows is outside the cache layout")
    if validation_contract != cache.layout.inference_contract_fingerprint:
        raise ValueError("validation inference contract fingerprint mismatch before routing")
    contract = _routing_contract(
        cache,
        preprocessor,
        calibration,
        config,
        np.asarray(attack_prior),
        validation_contract,
    )
    if cache.routing_contract_fingerprint not in (None, contract):
        raise ValueError("routing contract fingerprint mismatch")
    labels = list(cache.layout.routed_class_labels)
    router = CascadeStreamRouter(labels, calibration, dict(config), attack_prior)

    committed = cache.routed_committed_rows
    for start in range(0, committed, chunk_rows):
        end = min(start + chunk_rows, committed)
        positions = np.asarray(
            cache.arrays["routing_position"][start:end], dtype=np.int64
        )
        router.replay_batch(
            _metadata(cache, positions),
            np.asarray(cache.arrays["routed_probabilities"][positions]),
        )

    temporal = bool(config["temporal"].get("enabled", False)) and bool(
        config["cascade"].get("use_temporal_state", True)
    )
    run_paths: list[Path] = []
    order: Iterator[int] = iter(())
    try:
        if temporal:
            use_timestamp = _all_present_timestamps(cache, chunk_rows)
            run_paths = _external_order_runs(
                cache, Path(work_dir), chunk_rows, use_timestamp=use_timestamp
            )
            run_paths = _compact_order_runs(
                cache,
                run_paths,
                Path(work_dir),
                use_timestamp=use_timestamp,
                buffer_rows=chunk_rows,
            )
            order = _merge_order_runs(cache, run_paths, use_timestamp=use_timestamp)
        else:
            order = iter(range(cache.layout.row_count))
        for _ in range(committed):
            try:
                next(order)
            except StopIteration as error:
                raise RuntimeError("routing prefix exceeds validation order") from error
        while cache.routed_committed_rows < cache.layout.row_count:
            if (
                stop_after_routed_rows is not None
                and cache.routed_committed_rows >= stop_after_routed_rows
            ):
                break
            limit = chunk_rows
            if stop_after_routed_rows is not None:
                limit = min(limit, stop_after_routed_rows - cache.routed_committed_rows)
            positions_list: list[int] = []
            for _ in range(limit):
                try:
                    positions_list.append(next(order))
                except StopIteration:
                    break
            if not positions_list:
                break
            positions = np.asarray(positions_list, dtype=np.int64)
            known = np.asarray(cache.arrays["known_probabilities"][positions])
            selected = np.asarray(cache.arrays["selected_values"][positions])
            _, _, active_plus_unknown = preprocessor.apply_open_set(known, selected)
            main: np.ndarray[Any, Any] = np.zeros(
                (len(positions), len(labels)), dtype=np.float32
            )
            for column, label in enumerate([*preprocessor.active_labels, "unknown_like"]):
                main[:, labels.index(label)] = active_plus_unknown[:, column]
            main /= np.maximum(main.sum(axis=1, keepdims=True), 1e-12)
            flags = np.asarray(cache.arrays["boolean_flags"][positions], dtype=bool)
            boolean_fast_path = (
                flags.all(axis=1)
                if len(cache.layout.boolean_features)
                else np.zeros(len(positions), dtype=bool)
            )
            routed, stages, _ = router.route_batch(
                _metadata(cache, positions),
                np.asarray(cache.arrays["tiny_benign_probability"][positions]),
                main,
                boolean_fast_path,
            )
            cache.commit_routing_range(
                cache.routed_committed_rows,
                {
                    "cache_position": positions,
                    "routed_probabilities": routed,
                    "exit_stage": stages.astype(np.int16),
                },
                routing_contract_fingerprint=contract,
            )
        return contract
    finally:
        close = getattr(order, "close", None)
        if close is not None:
            close()
        for path in Path(work_dir).glob("order-*.npy"):
            path.unlink(missing_ok=True)


def calibrate_fixed_fpr_from_cache(
    cache: CalibrationCache,
    target_fprs: Sequence[float],
    work_dir: Path | str,
    *,
    chunk_rows: int = 65_536,
) -> dict[float, float]:
    """Calibrate exact fixed-FPR thresholds from routed benign validation rows."""

    _require_complete_inference(cache)
    if cache.routed_committed_rows != cache.layout.row_count:
        raise ValueError("fixed-FPR calibration requires complete routed validation")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    benign_label = cache.layout.true_class_labels.index("benign")
    benign_column = cache.layout.routed_class_labels.index("benign")
    count = 0
    for start in range(0, cache.layout.row_count, chunk_rows):
        end = min(start + chunk_rows, cache.layout.row_count)
        count += int(
            np.count_nonzero(cache.arrays["true_label"][start:end] == benign_label)
        )
    if not count:
        raise ValueError("fixed-FPR calibration requires benign validation examples")
    score_path = Path(work_dir) / "benign-attack-score.f64"
    scores: np.memmap[Any, Any] | None = None
    try:
        scores = _temporary_memmap(score_path, np.dtype("<f8"), (count,))
        offset = 0
        for start in range(0, cache.layout.row_count, chunk_rows):
            end = min(start + chunk_rows, cache.layout.row_count)
            mask = np.asarray(cache.arrays["true_label"][start:end]) == benign_label
            probabilities = np.asarray(
                cache.arrays["routed_probabilities"][start:end, benign_column],
                dtype=np.float64,
            )
            selected = 1.0 - probabilities[mask]
            scores[offset : offset + len(selected)] = selected
            offset += len(selected)
        thresholds: dict[float, float] = {}
        for raw in target_fprs:
            target = float(raw)
            if not 0.0 < target < 1.0:
                raise ValueError("target FPR values must be between 0 and 1")
            allowed = int(math.floor(target * count + 1e-12))
            if allowed == 0:
                threshold = np.nextafter(float(scores.max()), np.inf)
            else:
                index = count - allowed
                scores.partition(index)
                threshold = float(scores[index])
                tied_or_higher = 0
                for start in range(0, count, chunk_rows):
                    end = min(start + chunk_rows, count)
                    tied_or_higher += int(
                        np.count_nonzero(np.asarray(scores[start:end]) >= threshold)
                    )
                if tied_or_higher > allowed:
                    threshold = np.nextafter(threshold, np.inf)
            thresholds[target] = float(threshold)
        return thresholds
    finally:
        _close_memmap(scores)
        score_path.unlink(missing_ok=True)
