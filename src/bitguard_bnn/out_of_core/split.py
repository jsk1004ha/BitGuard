from __future__ import annotations

import hashlib
import heapq
import math
import os
import shutil
import sqlite3
import uuid
from collections import Counter, defaultdict
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from bitguard_bnn.constants import normalize_token
from bitguard_bnn.out_of_core.common import normalize_logical_path
from bitguard_bnn.out_of_core.manifest import (
    FileIdentity,
    SPLIT_MANIFEST_SCHEMA,
    SplitPlan,
    SourceRowRecord,
    attach_cleanup_context,
    canonical_json_bytes,
    file_identity,
    manifest_path_for_membership,
    split_manifest_semantic_fingerprint,
    split_manifest_semantics,
    stable_fingerprint,
    unlink_file_if_identity,
    write_json_atomic,
)
from bitguard_bnn.out_of_core.source import NormalizedChunk


INSPECTION_ALGORITHM = "bitguard.split-inspection.v1"
SPLIT_ALGORITHM = "bitguard.exact-split.v1"
RANDOM_RANK_ALGORITHM = "bitguard.keyed-blake2b-rank.v1"
STRATIFIED_QUOTA_ALGORITHM = "bitguard.hamilton-stratified-quota.v1"
SOURCE_MANIFEST_ALGORITHM = "bitguard.split-source-manifest.v1"
MEMBERSHIP_ALGORITHM = "bitguard.split-membership.v1"

_PARTITIONS = ("train", "validation", "test")
_REQUIRED_COLUMNS = {
    "row_uid",
    "source_file",
    "sequence_index",
    "behavior_label",
    "raw_attack",
    "device_id",
    "timestamp",
}

_INSPECTION_SCHEMA = pa.schema(
    [
        pa.field("row_uid", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_row", pa.int64(), nullable=False),
        pa.field("behavior_label", pa.string(), nullable=False),
        pa.field("raw_attack", pa.string(), nullable=False),
        pa.field("device_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=True),
    ]
)

_ASSIGNMENT_SCHEMA = pa.schema(
    [
        pa.field("row_uid", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("behavior_label", pa.string(), nullable=False),
        pa.field("original_behavior_label", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("source_row", pa.int64(), nullable=False),
        pa.field("raw_attack", pa.string(), nullable=False),
        pa.field("device_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=True),
    ]
)

_MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("row_uid", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("behavior_label", pa.string(), nullable=False),
    ]
)


class _ResourceTracker:
    def __init__(
        self,
        root: Path,
        max_rows_per_run: int,
        merge_read_batch_rows: int,
    ) -> None:
        self.root = root
        self.configured_max_rows_per_run = max_rows_per_run
        self.configured_merge_read_batch_rows = merge_read_batch_rows
        self.max_run_rows = 0
        self.run_count = 0
        self.temporary_bytes_peak = 0
        self.max_merge_fan_in_observed = 0
        self._merge_input_rows_buffered = 0
        self.max_merge_input_rows_buffered = 0

    def record_run(self, rows: int) -> None:
        self.max_run_rows = max(self.max_run_rows, rows)
        self.run_count += 1
        self.observe_disk()

    def observe_disk(self) -> None:
        size = 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file():
                    size += path.stat().st_size
            except OSError:
                continue
        self.temporary_bytes_peak = max(self.temporary_bytes_peak, size)

    def open_merge_batch(self, rows: int) -> None:
        self._merge_input_rows_buffered += rows
        self.max_merge_input_rows_buffered = max(
            self.max_merge_input_rows_buffered,
            self._merge_input_rows_buffered,
        )

    def close_merge_batch(self, rows: int) -> None:
        self._merge_input_rows_buffered -= rows
        if self._merge_input_rows_buffered < 0:
            raise RuntimeError("merge input row accounting underflow")


def _record_to_dict(record: SourceRowRecord) -> dict[str, Any]:
    return {
        "row_uid": record.row_uid,
        "source_file": record.source_file,
        "source_row": record.source_row,
        "behavior_label": record.behavior_label,
        "raw_attack": record.raw_attack,
        "device_id": record.device_id,
        "timestamp": record.timestamp,
    }


def _membership_record(
    row: Mapping[str, Any], split: str, *, effective_label: str | None = None
) -> dict[str, Any]:
    original = str(row["behavior_label"])
    return {
        "row_uid": str(row["row_uid"]),
        "split": split,
        "behavior_label": effective_label or original,
        "original_behavior_label": original,
        "source_file": str(row["source_file"]),
        "source_row": int(row["source_row"]),
        "raw_attack": str(row["raw_attack"]),
        "device_id": str(row["device_id"]),
        "timestamp": row["timestamp"],
    }


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("split", config)
    if not isinstance(raw, Mapping):
        raise ValueError("split configuration must be a mapping")
    strategy = str(raw.get("strategy", ""))
    if strategy not in {"random", "device", "attack", "time"}:
        raise ValueError(f"unsupported out-of-core split strategy: {strategy}")
    try:
        train = float(raw["train_fraction"])
        validation = float(raw["validation_fraction"])
        test = float(raw["test_fraction"])
        seed = int(raw["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("split fractions and seed are required") from exc
    if (
        any(
            isinstance(raw.get(key), bool)
            for key in (
                "train_fraction",
                "validation_fraction",
                "test_fraction",
                "seed",
            )
        )
        or min(train, validation, test) <= 0.0
        or not math.isclose(train + validation + test, 1.0, abs_tol=1e-9)
    ):
        raise ValueError("split fractions must be positive and sum to 1.0")
    devices = tuple(sorted({str(item) for item in raw.get("held_out_devices", [])}))
    attacks = tuple(
        sorted({normalize_token(item) for item in raw.get("held_out_attacks", [])})
    )
    return {
        "strategy": strategy,
        "train_fraction": train,
        "validation_fraction": validation,
        "test_fraction": test,
        "seed": seed,
        "held_out_devices": devices,
        "held_out_attacks": attacks,
    }


def _row_from_values(values: Mapping[str, Any], *, strategy: str) -> SourceRowRecord:
    uid = str(values["row_uid"])
    if not uid:
        raise ValueError("row_uid must be non-empty")
    source_file = normalize_logical_path(str(values["source_file"]))
    raw_source_row = values["sequence_index"]
    try:
        source_row = int(raw_source_row)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("source row must be an integer") from exc
    if (
        isinstance(raw_source_row, bool)
        or not isinstance(raw_source_row, Integral)
        or source_row < 0
    ):
        raise ValueError("source row must be a non-negative integer")
    label = str(values["behavior_label"])
    raw_attack = str(values["raw_attack"])
    device = str(values["device_id"])
    if not label or not raw_attack or not device:
        raise ValueError("split metadata values must be non-empty")
    raw_timestamp = values["timestamp"]
    timestamp: float | None
    try:
        timestamp = None if raw_timestamp is None else float(raw_timestamp)
    except (TypeError, ValueError, OverflowError):
        timestamp = None
    if timestamp is not None and not math.isfinite(timestamp):
        timestamp = None
    if strategy == "time" and timestamp is None:
        raise ValueError("time split does not accept missing or non-finite timestamps")
    return SourceRowRecord(
        row_uid=uid,
        source_file=source_file,
        source_row=source_row,
        behavior_label=label,
        raw_attack=raw_attack,
        device_id=device,
        timestamp=timestamp,
    )


def _fsync_file(path: Path) -> None:
    # Windows requires a writable handle for FlushFileBuffers/os.fsync.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _write_run(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
    tracker: _ResourceTracker,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(table, path, compression="zstd")
    _fsync_file(path)
    tracker.record_run(len(rows))


def _iter_run(
    path: Path,
    batch_rows: int,
    tracker: _ResourceTracker | None = None,
) -> Generator[dict[str, Any], None, None]:
    with path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(batch_size=batch_rows):
            rows = batch.to_pylist()
            row_count = len(rows)
            if tracker is not None:
                tracker.open_merge_batch(row_count)
            try:
                yield from rows
            finally:
                rows.clear()
                if tracker is not None:
                    tracker.close_merge_batch(row_count)
                del rows


def _merge_direct(
    paths: Sequence[Path],
    key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    merge_read_batch_rows: int,
    tracker: _ResourceTracker,
) -> Iterator[dict[str, Any]]:
    tracker.max_merge_fan_in_observed = max(
        tracker.max_merge_fan_in_observed, len(paths)
    )
    iterators = [_iter_run(path, merge_read_batch_rows, tracker) for path in paths]
    merged = heapq.merge(*iterators, key=key)
    try:
        yield from merged
    finally:
        for iterator in iterators:
            iterator.close()


def _write_merged_run(
    paths: Sequence[Path],
    destination: Path,
    key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    schema: pa.Schema,
    max_rows_per_run: int,
    merge_read_batch_rows: int,
    tracker: _ResourceTracker,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    buffer: list[dict[str, Any]] = []
    max_buffer_rows = 0
    try:
        for row in _merge_direct(paths, key, merge_read_batch_rows, tracker):
            buffer.append(row)
            if len(buffer) >= max_rows_per_run:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                max_buffer_rows = max(max_buffer_rows, len(buffer))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
            max_buffer_rows = max(max_buffer_rows, len(buffer))
    finally:
        writer.close()
    _fsync_file(destination)
    tracker.record_run(max_buffer_rows)


def _merge_runs(
    paths: Sequence[Path],
    key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    *,
    work: Path,
    schema: pa.Schema,
    max_rows_per_run: int,
    merge_read_batch_rows: int,
    merge_fan_in: int,
    tracker: _ResourceTracker,
) -> Iterator[dict[str, Any]]:
    current = list(paths)
    owned: set[Path] = set()
    merge_root = work / "merge" / uuid.uuid4().hex
    pass_number = 0
    try:
        while len(current) > merge_fan_in:
            previous_owned = set(owned)
            next_paths: list[Path] = []
            for group_index, start in enumerate(range(0, len(current), merge_fan_in)):
                group = current[start : start + merge_fan_in]
                if len(group) == 1:
                    next_paths.append(group[0])
                    continue
                destination = (
                    merge_root
                    / f"pass-{pass_number:03d}"
                    / f"run-{group_index:06d}.parquet"
                )
                _write_merged_run(
                    group,
                    destination,
                    key,
                    schema,
                    max_rows_per_run,
                    merge_read_batch_rows,
                    tracker,
                )
                owned.add(destination)
                next_paths.append(destination)
            carried = set(next_paths)
            for old_path in previous_owned - carried:
                old_path.unlink(missing_ok=True)
                owned.discard(old_path)
            current = next_paths
            pass_number += 1
        yield from _merge_direct(current, key, merge_read_batch_rows, tracker)
    finally:
        for path in owned:
            path.unlink(missing_ok=True)
        shutil.rmtree(merge_root, ignore_errors=True)


def _write_inspection_runs(
    chunks: Iterable[NormalizedChunk],
    work: Path,
    strategy: str,
    max_rows_per_run: int,
    tracker: _ResourceTracker,
) -> tuple[list[Path], list[Path], int]:
    uid_paths: list[Path] = []
    coordinate_paths: list[Path] = []
    buffer: list[dict[str, Any]] = []
    row_count = 0

    def flush() -> None:
        if not buffer:
            return
        index = len(uid_paths)
        uid_path = work / "inspection" / f"uid-{index:06d}.parquet"
        coordinate_path = work / "coordinates" / f"coordinate-{index:06d}.parquet"
        _write_run(
            uid_path,
            sorted(buffer, key=lambda row: str(row["row_uid"])),
            _INSPECTION_SCHEMA,
            tracker,
        )
        _write_run(
            coordinate_path,
            sorted(
                buffer,
                key=lambda row: (
                    str(row["source_file"]),
                    int(row["source_row"]),
                    str(row["row_uid"]),
                ),
            ),
            _INSPECTION_SCHEMA,
            tracker,
        )
        uid_paths.append(uid_path)
        coordinate_paths.append(coordinate_path)
        buffer.clear()

    for chunk in chunks:
        missing = _REQUIRED_COLUMNS - set(chunk.frame.columns)
        if missing:
            raise ValueError(f"normalized chunk is missing split columns: {sorted(missing)}")
        for values in chunk.frame[list(_REQUIRED_COLUMNS)].to_dict(orient="records"):
            buffer.append(_record_to_dict(_row_from_values(values, strategy=strategy)))
            row_count += 1
            if len(buffer) >= max_rows_per_run:
                flush()
    flush()
    if row_count == 0:
        raise ValueError("split input contains no rows")
    return uid_paths, coordinate_paths, row_count


def _validate_coordinates(
    paths: Sequence[Path],
    work: Path,
    max_rows_per_run: int,
    merge_read_batch_rows: int,
    merge_fan_in: int,
    tracker: _ResourceTracker,
) -> None:
    previous: tuple[str, int] | None = None
    for row in _merge_runs(
        paths,
        key=lambda item: (
            str(item["source_file"]),
            int(item["source_row"]),
            str(item["row_uid"]),
        ),
        work=work,
        schema=_INSPECTION_SCHEMA,
        max_rows_per_run=max_rows_per_run,
        merge_read_batch_rows=merge_read_batch_rows,
        merge_fan_in=merge_fan_in,
        tracker=tracker,
    ):
        coordinate = (str(row["source_file"]), int(row["source_row"]))
        if coordinate == previous:
            raise ValueError(f"duplicate logical source coordinate: {coordinate!r}")
        previous = coordinate


def _inspect_uid_order(
    paths: Sequence[Path],
    declared_source_fingerprint: str | None,
    configured_devices: set[str],
    configured_attacks: set[str],
    work: Path,
    max_rows_per_run: int,
    merge_read_batch_rows: int,
    merge_fan_in: int,
    tracker: _ResourceTracker,
) -> tuple[str, dict[str, Any]]:
    digest = hashlib.sha256()
    previous_uid: str | None = None
    source_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    max_device: str | None = None
    max_attack: str | None = None
    seen_devices: set[str] = set()
    seen_attacks: set[str] = set()
    for row in _merge_runs(
        paths,
        key=lambda item: (str(item["row_uid"]),),
        work=work,
        schema=_INSPECTION_SCHEMA,
        max_rows_per_run=max_rows_per_run,
        merge_read_batch_rows=merge_read_batch_rows,
        merge_fan_in=merge_fan_in,
        tracker=tracker,
    ):
        uid = str(row["row_uid"])
        if uid == previous_uid:
            raise ValueError(f"duplicate row_uid: {uid}")
        previous_uid = uid
        canonical = {
            "row_uid": uid,
            "source_file": str(row["source_file"]),
            "source_row": int(row["source_row"]),
            "behavior_label": str(row["behavior_label"]),
            "raw_attack": str(row["raw_attack"]),
            "device_id": str(row["device_id"]),
            "timestamp": row["timestamp"],
        }
        digest.update(canonical_json_bytes(canonical))
        digest.update(b"\n")
        source_counts[canonical["source_file"]] += 1
        label_counts[canonical["behavior_label"]] += 1
        device = canonical["device_id"]
        attack = normalize_token(canonical["raw_attack"])
        if device in configured_devices:
            seen_devices.add(device)
        if attack in configured_attacks:
            seen_attacks.add(attack)
        max_device = device if max_device is None or device > max_device else max_device
        if canonical["behavior_label"] != "benign":
            max_attack = attack if max_attack is None or attack > max_attack else max_attack
    derived = digest.hexdigest()
    source_fingerprint = stable_fingerprint(
        {
            "algorithm": SOURCE_MANIFEST_ALGORITHM,
            "declared_fingerprint": declared_source_fingerprint,
            "record_fingerprint": derived,
        }
    )
    return source_fingerprint, {
        "source_counts": source_counts,
        "label_counts": label_counts,
        "seen_devices": seen_devices,
        "seen_attacks": seen_attacks,
        "max_device": max_device,
        "max_attack": max_attack,
    }


def _random_priority(seed: int, row_uid: str) -> str:
    key = hashlib.sha256(str(int(seed)).encode("ascii")).digest()
    digest = hashlib.blake2b(
        digest_size=16,
        key=key,
        person=b"bitguard-split",
    )
    digest.update(row_uid.encode("utf-8"))
    return digest.hexdigest()


def _strategy_key(strategy: str, seed: int) -> Callable[[Mapping[str, Any]], tuple[Any, ...]]:
    if strategy == "time":
        return lambda row: (float(row["timestamp"]), str(row["row_uid"]))
    return lambda row: (
        str(row["behavior_label"]),
        _random_priority(seed, str(row["row_uid"])),
        str(row["row_uid"]),
    )


def _allocate_quota(counts: Mapping[str, int], target: int) -> dict[str, int]:
    total = sum(int(value) for value in counts.values())
    if target < 0 or target > total:
        raise ValueError("stratified quota target is outside the available rows")
    if total == 0:
        return {str(label): 0 for label in counts}
    allocation = {
        str(label): (int(count) * target) // total for label, count in counts.items()
    }
    remaining = target - sum(allocation.values())
    order = sorted(
        (str(label) for label in counts),
        key=lambda label: (-((int(counts[label]) * target) % total), label),
    )
    for label in order[:remaining]:
        allocation[label] += 1
    return allocation


def _partition_quotas(
    counts: Mapping[str, int], train_target: int, validation_target: int
) -> dict[str, dict[str, int]]:
    train = _allocate_quota(counts, train_target)
    remaining = {label: int(counts[label]) - train[label] for label in counts}
    validation = _allocate_quota(remaining, validation_target)
    test = {
        label: remaining[label] - validation[label]
        for label in remaining
    }
    return {"train": train, "validation": validation, "test": test}


def _prepare_strategy_runs(
    inspection_paths: Sequence[Path],
    work: Path,
    cfg: Mapping[str, Any],
    held_devices: set[str],
    held_attacks: set[str],
    max_rows_per_run: int,
    tracker: _ResourceTracker,
) -> tuple[list[Path], list[Path], Counter[str], int]:
    strategy = str(cfg["strategy"])
    seed = int(cfg["seed"])
    key = _strategy_key(strategy, seed)
    strategy_paths: list[Path] = []
    forced_paths: list[Path] = []
    strategy_buffer: list[dict[str, Any]] = []
    forced_buffer: list[dict[str, Any]] = []
    eligible_counts: Counter[str] = Counter()
    forced_count = 0

    def flush_strategy() -> None:
        if not strategy_buffer:
            return
        path = work / "strategy" / f"strategy-{len(strategy_paths):06d}.parquet"
        _write_run(path, sorted(strategy_buffer, key=key), _INSPECTION_SCHEMA, tracker)
        strategy_paths.append(path)
        strategy_buffer.clear()

    def flush_forced() -> None:
        if not forced_buffer:
            return
        path = work / "forced" / f"forced-{len(forced_paths):06d}.parquet"
        _write_run(
            path,
            sorted(forced_buffer, key=lambda row: str(row["row_uid"])),
            _ASSIGNMENT_SCHEMA,
            tracker,
        )
        forced_paths.append(path)
        forced_buffer.clear()

    for path in inspection_paths:
        for row in _iter_run(path, max_rows_per_run):
            forced = False
            if strategy == "device":
                forced = str(row["device_id"]) in held_devices
            elif strategy == "attack":
                forced = (
                    normalize_token(row["raw_attack"]) in held_attacks
                    or str(row["behavior_label"]) == "unknown_like"
                )
            if forced:
                forced_buffer.append(
                    _membership_record(
                        row,
                        "test",
                        effective_label="unknown_like" if strategy == "attack" else None,
                    )
                )
                forced_count += 1
                if len(forced_buffer) >= max_rows_per_run:
                    flush_forced()
            else:
                strategy_buffer.append(row)
                eligible_counts[str(row["behavior_label"])] += 1
                if len(strategy_buffer) >= max_rows_per_run:
                    flush_strategy()
        path.unlink()
    flush_strategy()
    flush_forced()
    return strategy_paths, forced_paths, eligible_counts, forced_count


def _assign_strategy_rows(
    strategy_paths: Sequence[Path],
    work: Path,
    cfg: Mapping[str, Any],
    eligible_counts: Mapping[str, int],
    forced_paths: Sequence[Path],
    forced_count: int,
    total_rows: int,
    max_rows_per_run: int,
    merge_read_batch_rows: int,
    merge_fan_in: int,
    tracker: _ResourceTracker,
) -> tuple[list[Path], dict[str, Any]]:
    strategy = str(cfg["strategy"])
    eligible = sum(eligible_counts.values())
    boundaries: dict[str, Any] = {}
    if strategy in {"time", "random"}:
        train_target = math.floor(total_rows * float(cfg["train_fraction"]))
        validation_target = math.floor(total_rows * float(cfg["validation_fraction"]))
    elif strategy == "device":
        validation_target = math.floor(
            eligible
            * float(cfg["validation_fraction"])
            / (float(cfg["train_fraction"]) + float(cfg["validation_fraction"]))
        )
        train_target = eligible - validation_target
    else:
        train_target = math.floor(eligible * float(cfg["train_fraction"]))
        validation_target = math.floor(eligible * float(cfg["validation_fraction"]))

    quotas = (
        None
        if strategy == "time"
        else _partition_quotas(eligible_counts, train_target, validation_target)
    )
    assignment_paths = list(forced_paths)
    buffer: list[dict[str, Any]] = []
    label_positions: Counter[str] = Counter()
    rank = 0
    key = _strategy_key(strategy, int(cfg["seed"]))

    def flush() -> None:
        if not buffer:
            return
        path = work / "assignments" / f"assignment-{len(assignment_paths):06d}.parquet"
        _write_run(
            path,
            sorted(buffer, key=lambda row: str(row["row_uid"])),
            _ASSIGNMENT_SCHEMA,
            tracker,
        )
        assignment_paths.append(path)
        buffer.clear()

    for row in _merge_runs(
        strategy_paths,
        key=key,
        work=work,
        schema=_INSPECTION_SCHEMA,
        max_rows_per_run=max_rows_per_run,
        merge_read_batch_rows=merge_read_batch_rows,
        merge_fan_in=merge_fan_in,
        tracker=tracker,
    ):
        if strategy == "time":
            if rank == train_target:
                boundaries["validation_start"] = {
                    "timestamp": float(row["timestamp"]),
                    "row_uid": str(row["row_uid"]),
                }
            if rank == train_target + validation_target:
                boundaries["test_start"] = {
                    "timestamp": float(row["timestamp"]),
                    "row_uid": str(row["row_uid"]),
                }
            split = (
                "train"
                if rank < train_target
                else "validation"
                if rank < train_target + validation_target
                else "test"
            )
        else:
            assert quotas is not None
            label = str(row["behavior_label"])
            position = label_positions[label]
            train_limit = quotas["train"][label]
            validation_limit = train_limit + quotas["validation"][label]
            split = (
                "train"
                if position < train_limit
                else "validation"
                if position < validation_limit
                else "test"
            )
            label_positions[label] += 1
        buffer.append(_membership_record(row, split))
        rank += 1
        if len(buffer) >= max_rows_per_run:
            flush()
    flush()
    for path in strategy_paths:
        path.unlink(missing_ok=True)
    boundaries["targets"] = {
        "eligible_rows": eligible,
        "forced_test_rows": forced_count,
        "train": train_target,
        "validation": validation_target,
        "eligible_test": eligible - train_target - validation_target,
    }
    if quotas is not None:
        boundaries["stratified_quotas"] = quotas
    return assignment_paths, boundaries


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_membership(
    assignment_paths: Sequence[Path],
    destination: Path,
    max_rows_per_run: int,
    merge_read_batch_rows: int,
    merge_fan_in: int,
    tracker: _ResourceTracker,
) -> tuple[dict[str, int], dict[str, Any], str, str]:
    counts = Counter({name: 0 for name in _PARTITIONS})
    class_counts: dict[str, Counter[str]] = {
        name: Counter() for name in _PARTITIONS
    }
    source_coverage: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_in_train = 0
    membership_digest = hashlib.sha256()
    previous_uid: str | None = None
    writer = pq.ParquetWriter(destination, _MEMBERSHIP_SCHEMA, compression="zstd")
    device_index_path = destination.with_name("device-overlap.sqlite")
    device_index = sqlite3.connect(device_index_path)
    device_index.execute(
        "CREATE TABLE membership (device_id TEXT NOT NULL, split TEXT NOT NULL, "
        "PRIMARY KEY (device_id, split)) WITHOUT ROWID"
    )
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=_MEMBERSHIP_SCHEMA))
            buffer.clear()

    try:
        try:
            for row in _merge_runs(
                assignment_paths,
                key=lambda item: (str(item["row_uid"]),),
                work=destination.parent,
                schema=_ASSIGNMENT_SCHEMA,
                max_rows_per_run=max_rows_per_run,
                merge_read_batch_rows=merge_read_batch_rows,
                merge_fan_in=merge_fan_in,
                tracker=tracker,
            ):
                uid = str(row["row_uid"])
                if uid == previous_uid:
                    raise RuntimeError(f"split membership overlap for row_uid: {uid}")
                previous_uid = uid
                split = str(row["split"])
                if split not in _PARTITIONS:
                    raise RuntimeError(f"invalid split membership: {split}")
                published = {name: row[name] for name in _MEMBERSHIP_SCHEMA.names}
                membership_digest.update(canonical_json_bytes(published))
                membership_digest.update(b"\n")
                counts[split] += 1
                class_counts[split][str(row["behavior_label"])] += 1
                source_coverage[str(row["source_file"])][split] += 1
                device_index.execute(
                    "INSERT OR IGNORE INTO membership (device_id, split) VALUES (?, ?)",
                    (str(row["device_id"]), split),
                )
                if split == "train" and str(row["behavior_label"]) == "unknown_like":
                    unknown_in_train += 1
                buffer.append(published)
                if len(buffer) >= max_rows_per_run:
                    flush()
            flush()
        finally:
            writer.close()
        device_index.commit()
        overlap = [
            str(row[0])
            for row in device_index.execute(
                "SELECT train.device_id FROM membership AS train "
                "JOIN membership AS test ON train.device_id = test.device_id "
                "WHERE train.split = 'train' AND test.split = 'test' "
                "ORDER BY train.device_id LIMIT 100"
            )
        ]
    finally:
        device_index.close()
    tracker.observe_disk()
    _fsync_file(destination)
    tracker.observe_disk()
    with destination.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if parquet.metadata.num_rows != sum(counts.values()):
            raise RuntimeError("split membership Parquet row count validation failed")
        if parquet.schema_arrow != _MEMBERSHIP_SCHEMA:
            raise RuntimeError("split membership Parquet schema validation failed")
    checks = {
        "row_uid_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "device_overlap": {
            "train_test": overlap
        },
        "unknown_in_train": unknown_in_train,
    }
    details = {
        "class_counts": {
            split: dict(sorted(values.items()))
            for split, values in class_counts.items()
        },
        "source_coverage": {
            source: {split: values.get(split, 0) for split in _PARTITIONS}
            for source, values in sorted(source_coverage.items())
        },
        "checks": checks,
    }
    return dict(counts), details, membership_digest.hexdigest(), _sha256_file(destination)


def _ensure_nonempty(counts: Mapping[str, int]) -> None:
    if any(int(counts.get(name, 0)) <= 0 for name in _PARTITIONS):
        raise ValueError("split must produce non-empty train, validation, and test partitions")


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _validate_existing(
    membership_path: Path,
    manifest_path: Path,
    expected_manifest: Mapping[str, Any],
) -> bool:
    if not membership_path.exists() and not manifest_path.exists():
        return False
    if not membership_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("incomplete immutable split output already exists")
    import json

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("split manifest root must be an object")
        payload_semantics = split_manifest_semantics(payload)
        payload_semantic_fingerprint = split_manifest_semantic_fingerprint(payload)
        expected_semantics = split_manifest_semantics(expected_manifest)
        expected_semantic_fingerprint = split_manifest_semantic_fingerprint(
            expected_manifest
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("immutable split output semantic conflict") from exc
    if (
        payload.get("semantic_fingerprint") != payload_semantic_fingerprint
        or expected_manifest.get("semantic_fingerprint")
        != expected_semantic_fingerprint
        or canonical_json_bytes(payload_semantics)
        != canonical_json_bytes(expected_semantics)
    ):
        raise RuntimeError("immutable split output semantic conflict")
    expected_rows = sum(
        int(value) for value in expected_manifest["counts"].values()
    )
    with membership_path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if (
            parquet.metadata.num_rows != expected_rows
            or parquet.schema_arrow != _MEMBERSHIP_SCHEMA
        ):
            raise RuntimeError("existing split membership validation failed")
    try:
        expected_sha = str(payload["membership"]["sha256"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError("immutable split output semantic conflict") from exc
    if _sha256_file(membership_path) != expected_sha:
        raise RuntimeError("existing split membership checksum mismatch")
    return True


def build_split_plan(
    chunks: Iterable[NormalizedChunk],
    config: Mapping[str, Any],
    output_dir: Path | str,
    *,
    max_rows_per_run: int = 65_536,
    merge_fan_in: int = 32,
    merge_read_batch_rows: int = 1_024,
    source_manifest_fingerprint: str | None = None,
) -> SplitPlan:
    """Inspect normalized chunks and publish an exact, UID-sorted split plan."""

    if isinstance(max_rows_per_run, bool) or int(max_rows_per_run) <= 0:
        raise ValueError("max_rows_per_run must be positive")
    max_rows_per_run = int(max_rows_per_run)
    if isinstance(merge_fan_in, bool) or int(merge_fan_in) < 2:
        raise ValueError("merge_fan_in must be at least 2")
    merge_fan_in = int(merge_fan_in)
    if (
        isinstance(merge_read_batch_rows, bool)
        or not isinstance(merge_read_batch_rows, Integral)
        or int(merge_read_batch_rows) <= 0
    ):
        raise ValueError("merge_read_batch_rows must be a positive integer")
    merge_read_batch_rows = int(merge_read_batch_rows)
    cfg = _validate_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    work = output / f".split-{uuid.uuid4().hex}.partial"
    work.mkdir()
    tracker = _ResourceTracker(
        work,
        max_rows_per_run,
        merge_read_batch_rows,
    )
    publication_partial: Path | None = None
    membership_path: Path | None = None
    manifest_path: Path | None = None
    published_membership: FileIdentity | None = None
    published_manifest: FileIdentity | None = None
    try:
        inspection_paths, coordinate_paths, total_rows = _write_inspection_runs(
            chunks,
            work,
            str(cfg["strategy"]),
            max_rows_per_run,
            tracker,
        )
        _validate_coordinates(
            coordinate_paths,
            work,
            max_rows_per_run,
            merge_read_batch_rows,
            merge_fan_in,
            tracker,
        )
        shutil.rmtree(work / "coordinates")
        configured_devices = set(cfg["held_out_devices"])
        configured_attacks = set(cfg["held_out_attacks"])
        source_fingerprint, inspection = _inspect_uid_order(
            inspection_paths,
            source_manifest_fingerprint,
            configured_devices,
            configured_attacks,
            work,
            max_rows_per_run,
            merge_read_batch_rows,
            merge_fan_in,
            tracker,
        )

        if configured_devices:
            absent = configured_devices - set(inspection["seen_devices"])
            if absent:
                raise ValueError(f"configured held-out devices are absent: {sorted(absent)}")
            held_devices = configured_devices
        elif cfg["strategy"] == "device":
            held_devices = {str(inspection["max_device"])}
        else:
            held_devices = set()
        if configured_attacks:
            absent = configured_attacks - set(inspection["seen_attacks"])
            if absent:
                raise ValueError(f"configured held-out attacks are absent: {sorted(absent)}")
            held_attacks = configured_attacks
        elif cfg["strategy"] == "attack":
            max_attack = inspection["max_attack"]
            if max_attack is None:
                raise ValueError("attack split requires at least one attack subtype")
            held_attacks = {str(max_attack)}
        else:
            held_attacks = set()

        strategy_paths, forced_paths, eligible_counts, forced_count = _prepare_strategy_runs(
            inspection_paths,
            work,
            cfg,
            held_devices,
            held_attacks,
            max_rows_per_run,
            tracker,
        )
        if cfg["strategy"] in {"device", "attack"} and (
            forced_count == 0 or sum(eligible_counts.values()) == 0
        ):
            raise ValueError("held-out group split must leave non-empty train and test sets")
        assignment_paths, boundaries = _assign_strategy_rows(
            strategy_paths,
            work,
            cfg,
            eligible_counts,
            forced_paths,
            forced_count,
            total_rows,
            max_rows_per_run,
            merge_read_batch_rows,
            merge_fan_in,
            tracker,
        )
        membership_partial = work / "membership.partial.parquet"
        counts, details, membership_digest, membership_sha = _write_membership(
            assignment_paths,
            membership_partial,
            max_rows_per_run,
            merge_read_batch_rows,
            merge_fan_in,
            tracker,
        )
        _ensure_nonempty(counts)
        if sum(counts.values()) != total_rows:
            raise RuntimeError("split membership is not exhaustive")
        for source, expected in inspection["source_counts"].items():
            actual = sum(int(value) for value in details["source_coverage"][source].values())
            if actual != int(expected):
                raise RuntimeError(f"split source coverage mismatch: {source}")
        if int(details["checks"]["unknown_in_train"]) != 0 and cfg["strategy"] == "attack":
            raise RuntimeError("attack split placed unknown_like rows in train")

        config_signature = stable_fingerprint(
            {
                "strategy": cfg["strategy"],
                "train_fraction": cfg["train_fraction"],
                "validation_fraction": cfg["validation_fraction"],
                "test_fraction": cfg["test_fraction"],
                "seed": cfg["seed"],
                "held_out_devices": list(cfg["held_out_devices"]),
                "held_out_attacks": list(cfg["held_out_attacks"]),
            }
        )
        fingerprint = stable_fingerprint(
            {
                "algorithm": MEMBERSHIP_ALGORITHM,
                "config_signature": config_signature,
                "membership_digest": membership_digest,
                "source_manifest_fingerprint": source_fingerprint,
            }
        )
        membership_path = output / f"split-membership-{fingerprint}.parquet"
        manifest_path = manifest_path_for_membership(membership_path)
        membership_schema = _schema_descriptor(_MEMBERSHIP_SCHEMA)
        inspection_schema = _schema_descriptor(_INSPECTION_SCHEMA)
        manifest: dict[str, Any] = {
            "schema_version": SPLIT_MANIFEST_SCHEMA,
            "fingerprint": fingerprint,
            "strategy": cfg["strategy"],
            "counts": {name: int(counts[name]) for name in _PARTITIONS},
            "class_counts": details["class_counts"],
            "source_coverage": details["source_coverage"],
            "source_manifest_fingerprint": source_fingerprint,
            "declared_source_manifest_fingerprint": source_manifest_fingerprint,
            "config_signature": config_signature,
            "checks": details["checks"],
            "rejections": {
                "missing_timestamp": 0,
                "nonfinite_timestamp": 0,
            },
            "inspection": {
                "rows": total_rows,
                "source_rows": dict(sorted(inspection["source_counts"].items())),
                "source_manifest_fingerprint": source_fingerprint,
            },
            "schema": membership_schema,
            "schemas": {
                "inspection": inspection_schema,
                "membership": membership_schema,
            },
            "schema_fingerprint": stable_fingerprint(membership_schema),
            "algorithm_versions": {
                "inspection": INSPECTION_ALGORITHM,
                "split": SPLIT_ALGORITHM,
                "random_rank": RANDOM_RANK_ALGORITHM,
                "stratified_quota": STRATIFIED_QUOTA_ALGORITHM,
                "source_manifest": SOURCE_MANIFEST_ALGORITHM,
                "membership": MEMBERSHIP_ALGORITHM,
            },
            "held_out": {
                "devices": sorted(held_devices),
                "attacks": sorted(held_attacks),
            },
            "ordering_boundaries": boundaries,
            "membership": {
                "path": membership_path.name,
                "sha256": membership_sha,
                "logical_digest": membership_digest,
                "rows": total_rows,
                "uid_sorted": True,
            },
            "resource_usage": {
                "configured_max_rows_per_run": max_rows_per_run,
                "max_run_rows": tracker.max_run_rows,
                "run_count": tracker.run_count,
                "merge_fan_in_limit": merge_fan_in,
                "max_merge_fan_in_observed": tracker.max_merge_fan_in_observed,
                "merge_read_batch_rows": tracker.configured_merge_read_batch_rows,
                "max_merge_input_rows_buffered": tracker.max_merge_input_rows_buffered,
                "merge_input_rows_buffered_limit": merge_fan_in * merge_read_batch_rows,
                "temporary_bytes_peak": tracker.temporary_bytes_peak,
                "final_membership_bytes": membership_partial.stat().st_size,
                "task1_verified_snapshot_disk_not_included": True,
            },
        }
        manifest["semantic_fingerprint"] = split_manifest_semantic_fingerprint(
            manifest
        )
        publication_partial = output / (
            f".{membership_path.name}.{uuid.uuid4().hex}.partial"
        )
        os.replace(membership_partial, publication_partial)
        shutil.rmtree(work)
        if _validate_existing(membership_path, manifest_path, manifest):
            publication_partial.unlink()
            publication_partial = None
            _fsync_directory(output)
            return SplitPlan(
                strategy=str(cfg["strategy"]),
                train_count=int(counts["train"]),
                validation_count=int(counts["validation"]),
                test_count=int(counts["test"]),
                membership_path=membership_path,
                fingerprint=fingerprint,
            )
        if membership_path.exists() or manifest_path.exists():
            raise RuntimeError("immutable split output appeared during publication")
        expected_membership_identity = file_identity(publication_partial)
        os.replace(publication_partial, membership_path)
        published_membership = expected_membership_identity
        publication_partial = None
        if file_identity(membership_path) != expected_membership_identity:
            raise RuntimeError("published membership identity changed after rename")
        _fsync_directory(output)
        published_manifest = write_json_atomic(manifest_path, manifest)
        return SplitPlan(
            strategy=str(cfg["strategy"]),
            train_count=int(counts["train"]),
            validation_count=int(counts["validation"]),
            test_count=int(counts["test"]),
            membership_path=membership_path,
            fingerprint=fingerprint,
        )
    except BaseException as primary:
        cleanup: list[BaseException] = []
        removed_final = False
        if published_manifest is not None and manifest_path is not None:
            try:
                removed_final = (
                    unlink_file_if_identity(manifest_path, published_manifest)
                    or removed_final
                )
            except BaseException as error:
                cleanup.append(error)
        if published_membership is not None and membership_path is not None:
            try:
                removed_final = (
                    unlink_file_if_identity(membership_path, published_membership)
                    or removed_final
                )
            except BaseException as error:
                cleanup.append(error)
        if publication_partial is not None:
            try:
                partial_existed = publication_partial.exists()
                publication_partial.unlink(missing_ok=True)
                removed_final = removed_final or partial_existed
            except BaseException as error:
                cleanup.append(error)
        if work.exists():
            try:
                shutil.rmtree(work)
            except BaseException as error:
                cleanup.append(error)
        if removed_final:
            try:
                _fsync_directory(output)
            except BaseException as error:
                cleanup.append(error)
        if cleanup:
            for cleanup_error in cleanup:
                attach_cleanup_context(
                    primary,
                    "cleanup failure: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
            raise primary from cleanup[0]
        raise


plan_split = build_split_plan


__all__ = [
    "INSPECTION_ALGORITHM",
    "MEMBERSHIP_ALGORITHM",
    "RANDOM_RANK_ALGORITHM",
    "SOURCE_MANIFEST_ALGORITHM",
    "SPLIT_ALGORITHM",
    "STRATIFIED_QUOTA_ALGORITHM",
    "build_split_plan",
    "plan_split",
]
