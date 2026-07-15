from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, cast

import pyarrow as pa
import pyarrow.parquet as pq

from bitguard_bnn.constants import normalize_token
from bitguard_bnn.out_of_core.manifest import (
    FileIdentity,
    SplitPlan,
    attach_cleanup_context,
    canonical_json_bytes,
    file_identity,
    read_split_manifest,
    split_manifest_semantic_fingerprint,
    stable_fingerprint,
    unlink_file_if_identity,
)
from bitguard_bnn.out_of_core.source import NormalizedChunk


SHARD_MANIFEST_SCHEMA = "bitguard.shard-manifest.v1"
SHARD_ALGORITHM = "bitguard.immutable-parquet-shards.v1"
COVERAGE_ALGORITHM = "bitguard.external-uid-coverage.v1"

_PARTITIONS = ("train", "validation", "test")
_PARTITION_SET = frozenset(_PARTITIONS)
_PATH_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_REQUIRED_SOURCE_COLUMNS = (
    "row_uid",
    "source_file",
    "sequence_index",
    "device_id",
    "raw_attack",
    "behavior_label",
    "timestamp",
)
_RESERVED_COLUMNS = frozenset((*_REQUIRED_SOURCE_COLUMNS, "dataset", "split"))
_MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("row_uid", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("behavior_label", pa.string(), nullable=False),
    ]
)
_COVERAGE_SCHEMA = _MEMBERSHIP_SCHEMA


@dataclass(frozen=True, slots=True)
class ShardPlan:
    dataset: str
    manifest_path: Path
    fingerprint: str
    row_count: int
    train_count: int
    validation_count: int
    test_count: int


@dataclass(slots=True)
class _ResourceTracker:
    root: Path
    max_rows_per_run: int
    max_run_rows: int = 0
    run_count: int = 0
    max_merge_fan_in_observed: int = 0
    temporary_bytes_peak: int = 0

    def record_run(self, rows: int) -> None:
        self.max_run_rows = max(self.max_run_rows, int(rows))
        self.run_count += 1
        self.observe_disk()

    def record_merge(self, fan_in: int) -> None:
        self.max_merge_fan_in_observed = max(
            self.max_merge_fan_in_observed, int(fan_in)
        )

    def observe_disk(self) -> None:
        total = 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        self.temporary_bytes_peak = max(self.temporary_bytes_peak, total)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _publish_file_no_replace(source: Path, destination: Path) -> FileIdentity:
    """Atomically publish a same-volume file without an overwrite race."""

    expected = file_identity(source)
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise RuntimeError(f"immutable artifact already exists: {destination}") from exc
    try:
        actual = file_identity(destination)
        if actual != expected:
            raise RuntimeError(f"published artifact identity changed: {destination}")
        source.unlink()
        _fsync_directory(destination.parent)
        return actual
    except BaseException as primary:
        cleanup: list[BaseException] = []
        try:
            unlink_file_if_identity(destination, expected)
        except BaseException as cleanup_failure:
            cleanup.append(cleanup_failure)
        if cleanup:
            for cleanup_error in cleanup:
                attach_cleanup_context(
                    primary,
                    "cleanup failure: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
            raise primary from cleanup[0]
        raise


def _write_manifest_no_replace(
    path: Path, payload: Mapping[str, Any]
) -> FileIdentity:
    partial = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    try:
        with partial.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(payload)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return _publish_file_no_replace(partial, path)
    except BaseException as primary:
        try:
            partial.unlink(missing_ok=True)
        except BaseException as cleanup:
            attach_cleanup_context(
                primary,
                f"cleanup failure: {type(cleanup).__name__}: {cleanup}",
            )
            raise primary from cleanup
        raise


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _shard_schema(selected_features: Sequence[str]) -> pa.Schema:
    fields = [
        pa.field("row_uid", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("sequence_index", pa.int64(), nullable=False),
        pa.field("device_id", pa.string(), nullable=False),
        pa.field("raw_attack", pa.string(), nullable=False),
        pa.field("behavior_label", pa.string(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=True),
    ]
    fields.extend(pa.field(name, pa.float32(), nullable=True) for name in selected_features)
    return pa.schema(fields)


def _working_schema(selected_features: Sequence[str]) -> pa.Schema:
    fields = list(_shard_schema(selected_features))
    fields.insert(7, pa.field("split", pa.string(), nullable=False))
    return pa.schema(fields)


def _validate_features(selected_features: Sequence[str]) -> tuple[str, ...]:
    features = tuple(str(name) for name in selected_features)
    if not features:
        raise ValueError("selected_features must not be empty")
    if any(not name for name in features):
        raise ValueError("selected_features must contain non-empty names")
    if len(set(features)) != len(features):
        raise ValueError("selected_features must not contain duplicates")
    collision = sorted(set(features) & _RESERVED_COLUMNS)
    if collision:
        raise ValueError(f"selected_features collide with metadata: {collision}")
    return features


def _validate_positive(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) < minimum:
        relation = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {relation}")
    return int(value)


def _write_run(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
    tracker: _ResourceTracker | None,
    row_group_rows: int = 1_024,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(records), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        row_group_size=min(len(records), row_group_rows),
    )
    _fsync_file(path)
    if tracker is not None:
        tracker.record_run(len(records))


def _iter_records(
    path: Path, batch_rows: int, columns: Sequence[str] | None = None
) -> Iterator[dict[str, Any]]:
    # Reopen per row group so an exception traceback cannot retain an open file
    # handle and prevent strict work-tree cleanup on Windows.
    with path.open("rb") as handle:
        row_groups = pq.ParquetFile(handle).metadata.num_row_groups
    for row_group in range(row_groups):
        with path.open("rb") as handle:
            table = pq.ParquetFile(handle).read_row_group(
                row_group, columns=columns
            )
        for batch in table.to_batches(max_chunksize=batch_rows):
            yield from batch.to_pylist()


def _write_merged_group(
    inputs: Sequence[Path],
    destination: Path,
    *,
    schema: pa.Schema,
    key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    output_batch_rows: int,
    read_batch_rows: int,
    tracker: _ResourceTracker | None,
) -> None:
    iterators = [_iter_records(path, read_batch_rows) for path in inputs]
    merged = heapq.merge(*iterators, key=key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(destination, schema, compression="zstd")
    buffer: list[dict[str, Any]] = []
    rows = 0
    try:
        for record in merged:
            buffer.append(record)
            if len(buffer) < output_batch_rows:
                continue
            writer.write_table(
                pa.Table.from_pylist(buffer, schema=schema),
                row_group_size=read_batch_rows,
            )
            rows += len(buffer)
            buffer.clear()
        if buffer:
            writer.write_table(
                pa.Table.from_pylist(buffer, schema=schema),
                row_group_size=read_batch_rows,
            )
            rows += len(buffer)
    except BaseException as primary:
        cleanup: list[BaseException] = []
        try:
            writer.close()
        except BaseException as cleanup_failure:
            cleanup.append(cleanup_failure)
        for iterator in iterators:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup_failure:
                    cleanup.append(cleanup_failure)
        if cleanup:
            for cleanup_error in cleanup:
                attach_cleanup_context(
                    primary,
                    "cleanup failure: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
            raise primary from cleanup[0]
        raise
    else:
        writer.close()
        for iterator in iterators:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
    _fsync_file(destination)
    if tracker is not None:
        tracker.record_merge(len(inputs))
        tracker.record_run(rows)


def _collapse_runs(
    paths: Sequence[Path],
    work: Path,
    *,
    schema: pa.Schema,
    key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    batch_rows: int,
    read_batch_rows: int,
    merge_fan_in: int,
    prefix: str,
    tracker: _ResourceTracker | None,
) -> Path:
    current = list(paths)
    if not current:
        raise ValueError("dataset contains no rows")
    pass_index = 0
    while len(current) > 1:
        next_paths: list[Path] = []
        for group_index, offset in enumerate(range(0, len(current), merge_fan_in)):
            group = current[offset : offset + merge_fan_in]
            if len(group) == 1:
                next_paths.append(group[0])
                continue
            destination = work / f"{prefix}-merge-{pass_index:04d}-{group_index:06d}.parquet"
            _write_merged_group(
                group,
                destination,
                schema=schema,
                key=key,
                output_batch_rows=batch_rows,
                read_batch_rows=read_batch_rows,
                tracker=tracker,
            )
            for path in group:
                path.unlink()
            next_paths.append(destination)
        current = next_paths
        pass_index += 1
    return current[0]


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"numeric value is not coercible: {value!r}") from exc
    if math.isnan(result):
        return None
    return result


def _exact_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        numeric = float(result)
    if not math.isfinite(numeric) or numeric != float(result):
        raise ValueError(f"{name} must be an integer")
    return result


def _source_key(row: Mapping[str, Any]) -> tuple[str]:
    return (str(row["row_uid"]),)


def _partition_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(row["split"]),
        str(row["behavior_label"]),
        str(row["source_file"]),
        int(row["sequence_index"]),
        str(row["row_uid"]),
    )


def _ordering_key(row: Mapping[str, Any]) -> list[Any]:
    return [
        str(row["source_file"]),
        int(row["sequence_index"]),
        str(row["row_uid"]),
    ]


def _write_source_runs(
    chunks: Iterable[NormalizedChunk],
    work: Path,
    *,
    selected_features: tuple[str, ...],
    schema: pa.Schema,
    max_rows_per_run: int,
    merge_read_rows: int,
    tracker: _ResourceTracker,
) -> list[Path]:
    runs: list[Path] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffer:
            return
        buffer.sort(key=_source_key)
        path = work / "source-runs" / f"run-{len(runs):08d}.parquet"
        _write_run(
            path, buffer, schema, tracker, row_group_rows=merge_read_rows
        )
        runs.append(path)
        buffer.clear()

    for chunk in chunks:
        frame = chunk.frame
        missing = [
            name
            for name in (*_REQUIRED_SOURCE_COLUMNS, *selected_features)
            if name not in frame.columns
        ]
        if missing:
            raise ValueError(f"normalized chunk is missing required columns: {missing}")
        columns = list(frame.columns)
        positions = {name: columns.index(name) for name in columns}
        for values in frame.itertuples(index=False, name=None):
            row_uid = str(values[positions["row_uid"]])
            source_file = str(values[positions["source_file"]])
            if not row_uid or not source_file:
                raise ValueError("row_uid and source_file must be non-empty")
            record: dict[str, Any] = {
                "row_uid": row_uid,
                "source_file": source_file,
                "sequence_index": _exact_int(
                    values[positions["sequence_index"]], name="sequence_index"
                ),
                "device_id": str(values[positions["device_id"]]),
                "raw_attack": str(values[positions["raw_attack"]]),
                "behavior_label": str(values[positions["behavior_label"]]),
                "timestamp": _finite_or_none(values[positions["timestamp"]]),
                "split": "train",
            }
            for feature in selected_features:
                record[feature] = _finite_or_none(values[positions[feature]])
            buffer.append(record)
            if len(buffer) >= max_rows_per_run:
                flush()
    flush()
    return runs


def _validate_membership_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"split membership is not a regular file: {path}")
    with path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if parquet.schema_arrow != _MEMBERSHIP_SCHEMA:
            raise RuntimeError("split membership schema drift")


def _validate_split_plan(plan: SplitPlan) -> dict[str, Any]:
    _validate_membership_file(plan.membership_path)
    try:
        manifest = read_split_manifest(plan)
        semantic = split_manifest_semantic_fingerprint(manifest)
        membership = manifest["membership"]
        counts = manifest["counts"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid split manifest for shard preparation") from exc
    expected_counts = {
        "train": plan.train_count,
        "validation": plan.validation_count,
        "test": plan.test_count,
    }
    if (
        manifest.get("fingerprint") != plan.fingerprint
        or manifest.get("semantic_fingerprint") != semantic
        or membership.get("path") != plan.membership_path.name
        or {name: int(counts.get(name, -1)) for name in _PARTITIONS}
        != expected_counts
        or int(membership.get("rows", -1)) != sum(expected_counts.values())
        or _sha256_file(plan.membership_path) != membership.get("sha256")
    ):
        raise RuntimeError("split plan fingerprint or membership checksum mismatch")
    return manifest


def _join_membership(
    source_path: Path,
    split_plan: SplitPlan,
    work: Path,
    *,
    schema: pa.Schema,
    max_rows_per_run: int,
    merge_read_rows: int,
    tracker: _ResourceTracker,
) -> tuple[list[Path], int, str]:
    _validate_membership_file(split_plan.membership_path)
    sources = _iter_records(source_path, merge_read_rows)
    members = _iter_records(split_plan.membership_path, merge_read_rows)
    source = next(sources, None)
    member = next(members, None)
    previous_source: str | None = None
    previous_member: str | None = None
    joined_runs: list[Path] = []
    buffer: list[dict[str, Any]] = []
    uid_digest = hashlib.sha256()
    total = 0

    def flush() -> None:
        if not buffer:
            return
        buffer.sort(key=_partition_key)
        path = work / "joined-runs" / f"run-{len(joined_runs):08d}.parquet"
        _write_run(
            path, buffer, schema, tracker, row_group_rows=merge_read_rows
        )
        joined_runs.append(path)
        buffer.clear()

    while source is not None or member is not None:
        if source is not None:
            source_uid = str(source["row_uid"])
            if source_uid == previous_source:
                raise ValueError(f"duplicate source row_uid: {source_uid}")
        else:
            source_uid = ""
        if member is not None:
            member_uid = str(member["row_uid"])
            if member_uid == previous_member:
                raise RuntimeError(f"duplicate split membership row_uid: {member_uid}")
        else:
            member_uid = ""
        if source is None:
            raise RuntimeError(f"missing source coverage for membership UID: {member_uid}")
        if member is None:
            raise RuntimeError(f"extra source coverage UID: {source_uid}")
        if source_uid < member_uid:
            raise RuntimeError(f"extra source coverage UID: {source_uid}")
        if source_uid > member_uid:
            raise RuntimeError(f"missing source coverage for membership UID: {member_uid}")
        split = str(member["split"])
        label = str(member["behavior_label"])
        if split not in _PARTITION_SET:
            raise RuntimeError(f"invalid split membership partition: {split}")
        if not _PATH_TOKEN.fullmatch(label):
            raise RuntimeError(f"unsafe behavior label for partition path: {label!r}")
        source_label = str(source["behavior_label"])
        sanctioned_attack_relabel = (
            split_plan.strategy == "attack"
            and split == "test"
            and label == "unknown_like"
        )
        if source_label != label and not sanctioned_attack_relabel:
            raise RuntimeError(
                "source and split membership behavior_label mismatch for UID: "
                f"{source_uid}"
            )
        published = dict(source)
        published["split"] = split
        published["behavior_label"] = label
        buffer.append(published)
        uid_digest.update(source_uid.encode("utf-8"))
        uid_digest.update(b"\n")
        total += 1
        previous_source = source_uid
        previous_member = member_uid
        source = next(sources, None)
        member = next(members, None)
        if len(buffer) >= max_rows_per_run:
            flush()
    flush()
    return joined_runs, total, uid_digest.hexdigest()


def _finalize_staged_shard(
    partial: Path,
    final: Path,
    *,
    schema: pa.Schema,
    rows: int,
) -> None:
    _fsync_file(partial)
    with partial.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if parquet.metadata.num_rows != rows:
            raise RuntimeError("staged shard row count validation failed")
        if parquet.schema_arrow != schema:
            raise RuntimeError("staged shard schema validation failed")
    os.replace(partial, final)
    _fsync_directory(final.parent)


def _write_staged_shards(
    sorted_path: Path,
    staging: Path,
    *,
    dataset: str,
    shard_schema: pa.Schema,
    shard_target_rows: int,
    record_batch_rows: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    part_numbers: Counter[tuple[str, str]] = Counter()
    writer: pq.ParquetWriter | None = None
    partial: Path | None = None
    final: Path | None = None
    current_bucket: tuple[str, str] | None = None
    rows = 0
    buffer: list[dict[str, Any]] = []
    uid_min: str | None = None
    uid_max: str | None = None
    ordering_min: list[Any] | None = None
    ordering_max: list[Any] | None = None
    sources: Counter[str] = Counter()

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            assert writer is not None
            writer.write_table(pa.Table.from_pylist(buffer, schema=shard_schema))
            buffer.clear()

    def close_shard() -> None:
        nonlocal writer, partial, final, rows, buffer
        nonlocal uid_min, uid_max, ordering_min, ordering_max, sources
        if writer is None:
            return
        flush_buffer()
        writer.close()
        writer = None
        assert partial is not None and final is not None and current_bucket is not None
        _finalize_staged_shard(partial, final, schema=shard_schema, rows=rows)
        relative = final.relative_to(staging).as_posix()
        split, label = current_bucket
        entries.append(
            {
                "path": relative,
                "sha256": _sha256_file(final),
                "byte_size": final.stat().st_size,
                "rows": rows,
                "split": split,
                "label": label,
                "label_counts": {label: rows},
                "schema_fingerprint": stable_fingerprint(_schema_descriptor(shard_schema)),
                "uid_min": uid_min,
                "uid_max": uid_max,
                "source_coverage": dict(sorted(sources.items())),
                "ordering_min": ordering_min,
                "ordering_max": ordering_max,
            }
        )
        partial = None
        final = None
        rows = 0
        buffer = []
        uid_min = None
        uid_max = None
        ordering_min = None
        ordering_max = None
        sources = Counter()

    def open_shard(bucket: tuple[str, str]) -> None:
        nonlocal writer, partial, final, current_bucket
        current_bucket = bucket
        split, label = bucket
        part = part_numbers[bucket]
        part_numbers[bucket] += 1
        directory = (
            staging
            / f"dataset={dataset}"
            / f"split={split}"
            / f"label={label}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"part-{part:08d}.parquet"
        partial = directory / f".part-{part:08d}.parquet.partial"
        writer = pq.ParquetWriter(partial, shard_schema, compression="zstd")

    try:
        for record in _iter_records(sorted_path, record_batch_rows):
            bucket = (str(record["split"]), str(record["behavior_label"]))
            if writer is None:
                open_shard(bucket)
            elif bucket != current_bucket or rows >= shard_target_rows:
                close_shard()
                open_shard(bucket)
            uid = str(record["row_uid"])
            key = _ordering_key(record)
            uid_min = uid if uid_min is None else min(uid_min, uid)
            uid_max = uid if uid_max is None else max(uid_max, uid)
            ordering_min = key if ordering_min is None else ordering_min
            ordering_max = key
            sources[str(record["source_file"])] += 1
            buffer.append(record)
            rows += 1
            if len(buffer) >= record_batch_rows or rows >= shard_target_rows:
                flush_buffer()
        close_shard()
    except BaseException as primary:
        if writer is not None:
            try:
                writer.close()
            except BaseException as cleanup:
                attach_cleanup_context(
                    primary,
                    f"cleanup failure: {type(cleanup).__name__}: {cleanup}",
                )
                raise primary from cleanup
        raise
    return sorted(entries, key=lambda entry: str(entry["path"]))


def _manifest_semantics(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema_version",
        "dataset",
        "preprocessing_fingerprint",
        "split_fingerprint",
        "selected_features",
        "counts",
        "class_counts",
        "source_coverage",
        "coverage",
        "shard_contract",
        "schema",
        "schema_fingerprint",
        "algorithm_versions",
        "entries",
    )
    try:
        return {name: payload[name] for name in fields}
    except (KeyError, TypeError) as exc:
        raise ValueError("shard manifest is missing semantic fields") from exc


def _build_manifest(
    *,
    dataset: str,
    preprocessing_fingerprint: str,
    split_plan: SplitPlan,
    selected_features: tuple[str, ...],
    schema: pa.Schema,
    entries: list[dict[str, Any]],
    row_count: int,
    uid_digest: str,
    tracker: _ResourceTracker,
    shard_target_rows: int,
    record_batch_rows: int,
    merge_fan_in: int,
    merge_read_rows: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_coverage: Counter[str] = Counter()
    for entry in entries:
        split = str(entry["split"])
        label = str(entry["label"])
        rows = int(entry["rows"])
        counts[split] += rows
        class_counts[split][label] += rows
        source_coverage.update(
            {str(name): int(value) for name, value in entry["source_coverage"].items()}
        )
    payload: dict[str, Any] = {
        "schema_version": SHARD_MANIFEST_SCHEMA,
        "dataset": dataset,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "split_fingerprint": split_plan.fingerprint,
        "selected_features": list(selected_features),
        "counts": {name: int(counts[name]) for name in _PARTITIONS},
        "class_counts": {
            name: dict(sorted(class_counts[name].items())) for name in _PARTITIONS
        },
        "source_coverage": dict(sorted(source_coverage.items())),
        "coverage": {"rows": row_count, "uid_digest": uid_digest},
        "shard_contract": {
            "shard_target_rows": shard_target_rows,
            "record_batch_rows": record_batch_rows,
            "max_rows_per_run": tracker.max_rows_per_run,
            "merge_fan_in": merge_fan_in,
            "merge_read_rows": merge_read_rows,
        },
        "schema": _schema_descriptor(schema),
        "schema_fingerprint": stable_fingerprint(_schema_descriptor(schema)),
        "algorithm_versions": {
            "shards": SHARD_ALGORITHM,
            "coverage": COVERAGE_ALGORITHM,
        },
        "entries": entries,
        "resource_usage": {
            "shard_target_rows": shard_target_rows,
            "record_batch_rows": record_batch_rows,
            "configured_max_rows_per_run": tracker.max_rows_per_run,
            "max_run_rows": tracker.max_run_rows,
            "run_count": tracker.run_count,
            "merge_fan_in_limit": merge_fan_in,
            "merge_read_rows": merge_read_rows,
            "max_merge_fan_in_observed": tracker.max_merge_fan_in_observed,
            "temporary_bytes_peak": tracker.temporary_bytes_peak,
        },
    }
    payload["fingerprint"] = stable_fingerprint(_manifest_semantics(payload))
    return payload


def load_shard_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read shard manifest: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SHARD_MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported shard manifest schema: {manifest_path}")
    try:
        fingerprint = stable_fingerprint(_manifest_semantics(payload))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid shard manifest semantics") from exc
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError("shard manifest fingerprint mismatch")
    return payload


def _safe_entry_path(root: Path, relative: object) -> Path:
    value = str(relative)
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise RuntimeError(f"unsafe shard manifest path: {value!r}")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"shard is not a regular file: {value}")
    return path


def _ensure_safe_directory(root: Path, directory: Path) -> None:
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"shard directory escapes output root: {directory}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise RuntimeError(f"unsafe shard destination directory: {current}")
            continue
        current.mkdir()
        if current.is_symlink() or not current.is_dir():
            raise RuntimeError(f"unsafe shard destination directory: {current}")


def _coverage_key(row: Mapping[str, Any]) -> tuple[str]:
    return (str(row["row_uid"]),)


def _write_coverage_run(
    records: list[dict[str, Any]],
    path: Path,
    tracker: _ResourceTracker | None,
    *,
    row_group_rows: int,
) -> None:
    records.sort(key=_coverage_key)
    _write_run(
        path,
        records,
        _COVERAGE_SCHEMA,
        tracker,
        row_group_rows=row_group_rows,
    )


def _verify_entry_and_index(
    path: Path,
    entry: Mapping[str, Any],
    *,
    schema: pa.Schema,
    work: Path,
    run_paths: list[Path],
    max_rows_per_run: int,
    merge_read_rows: int,
    tracker: _ResourceTracker,
) -> tuple[Counter[str], dict[str, Counter[str]], Counter[str]]:
    schema_fingerprint = stable_fingerprint(_schema_descriptor(schema))
    if entry.get("schema_fingerprint") != schema_fingerprint:
        raise RuntimeError(f"shard entry schema fingerprint mismatch: {entry.get('path')}")
    if _sha256_file(path) != str(entry.get("sha256")):
        raise RuntimeError(f"shard checksum mismatch: {entry.get('path')}")
    if path.stat().st_size != int(entry.get("byte_size", -1)):
        raise RuntimeError(f"shard byte size mismatch: {entry.get('path')}")
    with path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if parquet.schema_arrow != schema:
            raise RuntimeError(f"shard schema drift: {entry.get('path')}")
        if parquet.metadata.num_rows != int(entry.get("rows", -1)):
            raise RuntimeError(f"shard row count mismatch: {entry.get('path')}")
    expected_split = str(entry.get("split"))
    expected_label = str(entry.get("label"))
    counts: Counter[str] = Counter()
    classes: dict[str, Counter[str]] = defaultdict(Counter)
    sources: Counter[str] = Counter()
    entry_sources: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    uid_min: str | None = None
    uid_max: str | None = None
    ordering_min: list[Any] | None = None
    ordering_max: list[Any] | None = None
    coverage_buffer: list[dict[str, Any]] = []

    def flush() -> None:
        if not coverage_buffer:
            return
        run = work / "coverage-runs" / f"run-{len(run_paths):08d}.parquet"
        _write_coverage_run(
            coverage_buffer,
            run,
            tracker,
            row_group_rows=merge_read_rows,
        )
        run_paths.append(run)
        coverage_buffer.clear()

    for row in _iter_records(path, max_rows_per_run):
        split = expected_split
        label = str(row["behavior_label"])
        if label != expected_label:
            raise RuntimeError(f"shard partition metadata mismatch: {entry.get('path')}")
        uid = str(row["row_uid"])
        key = _ordering_key(row)
        if ordering_max is not None and tuple(key) < tuple(ordering_max):
            raise RuntimeError(f"shard ordering mismatch: {entry.get('path')}")
        ordering_min = key if ordering_min is None else ordering_min
        ordering_max = key
        uid_min = uid if uid_min is None else min(uid_min, uid)
        uid_max = uid if uid_max is None else max(uid_max, uid)
        source = str(row["source_file"])
        counts[split] += 1
        classes[split][label] += 1
        sources[source] += 1
        entry_sources[source] += 1
        label_counts[label] += 1
        coverage_buffer.append(
            {"row_uid": uid, "split": split, "behavior_label": label}
        )
        if len(coverage_buffer) >= max_rows_per_run:
            flush()
    flush()
    if dict(sorted(entry_sources.items())) != {
        str(name): int(value) for name, value in entry.get("source_coverage", {}).items()
    }:
        raise RuntimeError(f"shard source coverage mismatch: {entry.get('path')}")
    if dict(sorted(label_counts.items())) != {
        str(name): int(value) for name, value in entry.get("label_counts", {}).items()
    }:
        raise RuntimeError(f"shard label count mismatch: {entry.get('path')}")
    if (
        uid_min != entry.get("uid_min")
        or uid_max != entry.get("uid_max")
        or ordering_min != entry.get("ordering_min")
        or ordering_max != entry.get("ordering_max")
    ):
        raise RuntimeError(f"shard ordering boundaries mismatch: {entry.get('path')}")
    return counts, classes, sources


def _verify_uid_coverage(
    actual_path: Path,
    *,
    manifest: Mapping[str, Any],
    split_plan: SplitPlan | None,
    batch_rows: int,
) -> None:
    actual_iter = _iter_records(actual_path, batch_rows)
    expected_iter = (
        _iter_records(split_plan.membership_path, batch_rows)
        if split_plan is not None
        else None
    )
    previous: str | None = None
    digest = hashlib.sha256()
    rows = 0
    for actual in actual_iter:
        uid = str(actual["row_uid"])
        if uid == previous:
            raise RuntimeError(f"duplicate shard row_uid or split overlap: {uid}")
        if previous is not None and uid < previous:
            raise RuntimeError("coverage UID index is not sorted")
        digest.update(uid.encode("utf-8"))
        digest.update(b"\n")
        previous = uid
        rows += 1
        if expected_iter is not None:
            expected = next(expected_iter, None)
            if expected is None:
                raise RuntimeError(f"extra shard coverage UID: {uid}")
            expected_tuple = (
                str(expected["row_uid"]),
                str(expected["split"]),
                str(expected["behavior_label"]),
            )
            actual_tuple = (
                uid,
                str(actual["split"]),
                str(actual["behavior_label"]),
            )
            if actual_tuple != expected_tuple:
                if uid < expected_tuple[0]:
                    raise RuntimeError(f"extra shard coverage UID: {uid}")
                if uid > expected_tuple[0]:
                    raise RuntimeError(f"missing shard coverage UID: {expected_tuple[0]}")
                raise RuntimeError(f"shard split or label mismatch for UID: {uid}")
    if expected_iter is not None:
        remaining = next(expected_iter, None)
        if remaining is not None:
            raise RuntimeError(f"missing shard coverage UID: {remaining['row_uid']}")
    coverage = manifest.get("coverage", {})
    if rows != int(coverage.get("rows", -1)):
        raise RuntimeError("shard coverage row count mismatch")
    if digest.hexdigest() != coverage.get("uid_digest"):
        raise RuntimeError("shard coverage UID digest mismatch")


def _cleanup_work(work: Path, primary: BaseException | None = None) -> None:
    if not work.exists():
        return
    try:
        shutil.rmtree(work)
    except BaseException as cleanup:
        if primary is None:
            raise
        attach_cleanup_context(
            primary,
            f"cleanup failure: {type(cleanup).__name__}: {cleanup}",
        )
        raise primary from cleanup


def _reject_stale_work(root: Path) -> None:
    stale = sorted(
        (
            path
            for pattern in (".shards-*.partial", ".verify-shards-*.partial")
            for path in root.glob(pattern)
        ),
        key=lambda path: path.name,
    )
    if stale:
        raise RuntimeError(f"unsafe stale shard work artifact exists: {stale[0]}")


def verify_shard_manifest(
    manifest_path: Path | str,
    *,
    split_plan: SplitPlan | None = None,
    preprocessing_fingerprint: str | None = None,
    max_rows_per_run: int = 65_536,
    merge_fan_in: int = 32,
    merge_read_rows: int = 1_024,
) -> dict[str, Any]:
    """Verify immutable shard bytes, schemas, counts and exact UID coverage."""

    max_rows_per_run = _validate_positive("max_rows_per_run", max_rows_per_run)
    merge_fan_in = _validate_positive("merge_fan_in", merge_fan_in, minimum=2)
    merge_read_rows = _validate_positive("merge_read_rows", merge_read_rows)
    path = Path(manifest_path)
    manifest = load_shard_manifest(path)
    _reject_stale_work(path.parent)
    if preprocessing_fingerprint is not None and manifest.get(
        "preprocessing_fingerprint"
    ) != str(preprocessing_fingerprint):
        raise RuntimeError("preprocessing fingerprint mismatch")
    if split_plan is not None:
        if manifest.get("split_fingerprint") != split_plan.fingerprint:
            raise RuntimeError("split fingerprint mismatch")
        _validate_split_plan(split_plan)
    try:
        selected_features = _validate_features(manifest["selected_features"])
        schema = _shard_schema(selected_features)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid shard feature contract") from exc
    descriptor = _schema_descriptor(schema)
    if manifest.get("schema") != descriptor or manifest.get(
        "schema_fingerprint"
    ) != stable_fingerprint(descriptor):
        raise RuntimeError("shard manifest schema fingerprint mismatch")
    work = path.parent / f".verify-shards-{uuid.uuid4().hex}.partial"
    work.mkdir(parents=True)
    tracker = _ResourceTracker(work, max_rows_per_run)
    try:
        entries_value = manifest.get("entries")
        if not isinstance(entries_value, list) or not entries_value:
            raise RuntimeError("shard manifest has no entries")
        entries = sorted(entries_value, key=lambda item: str(item.get("path")))
        if entries != entries_value:
            raise RuntimeError("shard manifest entries are not path-sorted")
        listed: set[Path] = set()
        run_paths: list[Path] = []
        counts: Counter[str] = Counter()
        classes: dict[str, Counter[str]] = defaultdict(Counter)
        sources: Counter[str] = Counter()
        bucket_entries: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise RuntimeError("invalid shard manifest entry")
            shard = _safe_entry_path(path.parent, entry.get("path"))
            split = str(entry.get("split"))
            label = str(entry.get("label"))
            if split not in _PARTITION_SET or not _PATH_TOKEN.fullmatch(label):
                raise RuntimeError("invalid shard partition manifest entry")
            expected_relative = PurePosixPath(
                f"dataset={manifest['dataset']}",
                f"split={split}",
                f"label={label}",
                PurePosixPath(str(entry.get("path"))).name,
            ).as_posix()
            if str(entry.get("path")) != expected_relative:
                raise RuntimeError(f"shard path/partition mismatch: {entry.get('path')}")
            bucket_entries[(split, label)].append(entry)
            resolved = shard.resolve(strict=True)
            if resolved in listed:
                raise RuntimeError(f"duplicate shard manifest path: {entry.get('path')}")
            listed.add(resolved)
            entry_counts, entry_classes, entry_sources = _verify_entry_and_index(
                shard,
                entry,
                schema=schema,
                work=work,
                run_paths=run_paths,
                max_rows_per_run=max_rows_per_run,
                merge_read_rows=merge_read_rows,
                tracker=tracker,
            )
            counts.update(entry_counts)
            for split, values in entry_classes.items():
                classes[split].update(values)
            sources.update(entry_sources)
        target_rows = int(manifest.get("shard_contract", {}).get("shard_target_rows", -1))
        if target_rows <= 0:
            raise RuntimeError("invalid shard target row contract")
        for bucket, bucket_values in bucket_entries.items():
            previous_max: tuple[Any, ...] | None = None
            for index, entry in enumerate(bucket_values):
                expected_name = f"part-{index:08d}.parquet"
                if PurePosixPath(str(entry["path"])).name != expected_name:
                    raise RuntimeError(f"non-contiguous shard parts for bucket: {bucket}")
                rows = int(entry["rows"])
                if rows <= 0 or rows > target_rows:
                    raise RuntimeError(f"shard target row bound exceeded: {entry['path']}")
                if index < len(bucket_values) - 1 and rows != target_rows:
                    raise RuntimeError(f"non-final shard is below target rows: {entry['path']}")
                ordering_min = tuple(entry["ordering_min"])
                ordering_max = tuple(entry["ordering_max"])
                if previous_max is not None and ordering_min < previous_max:
                    raise RuntimeError(f"shard ordering overlap for bucket: {bucket}")
                previous_max = ordering_max
        actual_counts = {name: int(counts[name]) for name in _PARTITIONS}
        actual_classes = {
            name: dict(sorted(classes[name].items())) for name in _PARTITIONS
        }
        if actual_counts != manifest.get("counts"):
            raise RuntimeError("shard partition counts mismatch")
        if actual_classes != manifest.get("class_counts"):
            raise RuntimeError("shard class counts mismatch")
        if dict(sorted(sources.items())) != manifest.get("source_coverage"):
            raise RuntimeError("shard source coverage mismatch")
        if split_plan is not None and actual_counts != {
            "train": split_plan.train_count,
            "validation": split_plan.validation_count,
            "test": split_plan.test_count,
        }:
            raise RuntimeError("shard counts do not match split plan")
        sorted_coverage = _collapse_runs(
            run_paths,
            work / "coverage-merges",
            schema=_COVERAGE_SCHEMA,
            key=_coverage_key,
            batch_rows=max_rows_per_run,
            read_batch_rows=merge_read_rows,
            merge_fan_in=merge_fan_in,
            prefix="coverage",
            tracker=tracker,
        )
        _verify_uid_coverage(
            sorted_coverage,
            manifest=manifest,
            split_plan=split_plan,
            batch_rows=max_rows_per_run,
        )
        dataset_root = path.parent / f"dataset={manifest['dataset']}"
        if dataset_root.is_symlink() or not dataset_root.is_dir():
            raise RuntimeError("shard dataset directory is missing or unsafe")
        for partial in dataset_root.rglob("*.partial"):
            raise RuntimeError(f"unsafe partial shard artifact: {partial}")
        discovered = {
            candidate.resolve(strict=True)
            for candidate in dataset_root.rglob("*.parquet")
            if candidate.is_file() and not candidate.is_symlink()
        }
        if discovered != listed:
            raise RuntimeError("unlisted or missing Parquet shard artifacts")
        _cleanup_work(work)
        return manifest
    except BaseException as primary:
        _cleanup_work(work, primary)
        raise


def _plan_from_manifest(path: Path, manifest: Mapping[str, Any]) -> ShardPlan:
    counts = manifest["counts"]
    return ShardPlan(
        dataset=str(manifest["dataset"]),
        manifest_path=path,
        fingerprint=str(manifest["fingerprint"]),
        row_count=sum(int(counts[name]) for name in _PARTITIONS),
        train_count=int(counts["train"]),
        validation_count=int(counts["validation"]),
        test_count=int(counts["test"]),
    )


def write_parquet_shards(
    chunks: Iterable[NormalizedChunk],
    split_plan: SplitPlan,
    selected_features: Sequence[str],
    output_dir: Path | str,
    *,
    dataset_name: str,
    preprocessing_fingerprint: str,
    shard_target_rows: int = 1_000_000,
    record_batch_rows: int = 65_536,
    max_rows_per_run: int = 65_536,
    merge_fan_in: int = 32,
    merge_read_rows: int = 1_024,
) -> ShardPlan:
    """Build immutable selected-feature shards through bounded external merges."""

    features = _validate_features(selected_features)
    shard_target_rows = _validate_positive("shard_target_rows", shard_target_rows)
    record_batch_rows = _validate_positive("record_batch_rows", record_batch_rows)
    max_rows_per_run = _validate_positive("max_rows_per_run", max_rows_per_run)
    merge_fan_in = _validate_positive("merge_fan_in", merge_fan_in, minimum=2)
    merge_read_rows = _validate_positive("merge_read_rows", merge_read_rows)
    if not str(dataset_name).strip():
        raise ValueError("dataset_name must not be empty")
    dataset = normalize_token(dataset_name)
    if not _PATH_TOKEN.fullmatch(dataset):
        raise ValueError("dataset_name does not produce a safe partition token")
    preprocessing_fingerprint = str(preprocessing_fingerprint)
    if not preprocessing_fingerprint:
        raise ValueError("preprocessing_fingerprint must not be empty")
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan")
    _validate_split_plan(split_plan)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _reject_stale_work(output)
    work = output / f".shards-{uuid.uuid4().hex}.partial"
    work.mkdir()
    tracker = _ResourceTracker(work, max_rows_per_run)
    shard_schema = _shard_schema(features)
    working_schema = _working_schema(features)
    manifest_path = output / "shard_manifest.json"
    published: list[tuple[Path, FileIdentity]] = []
    published_manifest: FileIdentity | None = None
    try:
        source_runs = _write_source_runs(
            chunks,
            work,
            selected_features=features,
            schema=working_schema,
            max_rows_per_run=max_rows_per_run,
            merge_read_rows=merge_read_rows,
            tracker=tracker,
        )
        sorted_source = _collapse_runs(
            source_runs,
            work / "source-merges",
            schema=working_schema,
            key=_source_key,
            batch_rows=max_rows_per_run,
            read_batch_rows=merge_read_rows,
            merge_fan_in=merge_fan_in,
            prefix="source",
            tracker=tracker,
        )
        joined_runs, row_count, uid_digest = _join_membership(
            sorted_source,
            split_plan,
            work,
            schema=working_schema,
            max_rows_per_run=max_rows_per_run,
            merge_read_rows=merge_read_rows,
            tracker=tracker,
        )
        sorted_joined = _collapse_runs(
            joined_runs,
            work / "joined-merges",
            schema=working_schema,
            key=_partition_key,
            batch_rows=max_rows_per_run,
            read_batch_rows=merge_read_rows,
            merge_fan_in=merge_fan_in,
            prefix="joined",
            tracker=tracker,
        )
        staging = work / "staged"
        entries = _write_staged_shards(
            sorted_joined,
            staging,
            dataset=dataset,
            shard_schema=shard_schema,
            shard_target_rows=shard_target_rows,
            record_batch_rows=min(record_batch_rows, shard_target_rows),
        )
        tracker.observe_disk()
        manifest = _build_manifest(
            dataset=dataset,
            preprocessing_fingerprint=preprocessing_fingerprint,
            split_plan=split_plan,
            selected_features=features,
            schema=shard_schema,
            entries=entries,
            row_count=row_count,
            uid_digest=uid_digest,
            tracker=tracker,
            shard_target_rows=shard_target_rows,
            record_batch_rows=record_batch_rows,
            merge_fan_in=merge_fan_in,
            merge_read_rows=merge_read_rows,
        )
        if manifest_path.exists():
            # The expected semantic manifest is complete; discard the private
            # candidate before independently verifying an existing publication.
            _cleanup_work(work)
            existing = verify_shard_manifest(
                manifest_path,
                split_plan=split_plan,
                preprocessing_fingerprint=preprocessing_fingerprint,
                max_rows_per_run=max_rows_per_run,
                merge_fan_in=merge_fan_in,
                merge_read_rows=merge_read_rows,
            )
            if canonical_json_bytes(_manifest_semantics(existing)) != canonical_json_bytes(
                _manifest_semantics(manifest)
            ):
                raise RuntimeError("immutable shard output semantic conflict")
            return _plan_from_manifest(manifest_path, existing)
        final_dataset_root = output / f"dataset={dataset}"
        if final_dataset_root.exists() and (
            final_dataset_root.is_symlink()
            or not final_dataset_root.is_dir()
            or any(final_dataset_root.rglob("*"))
        ):
            raise RuntimeError("incomplete or unsafe immutable shard output already exists")
        for entry in entries:
            relative = PurePosixPath(str(entry["path"]))
            staged = staging.joinpath(*relative.parts)
            destination = output.joinpath(*relative.parts)
            _ensure_safe_directory(output, destination.parent)
            identity = _publish_file_no_replace(staged, destination)
            published.append((destination, identity))
        _cleanup_work(work)
        published_manifest = _write_manifest_no_replace(manifest_path, manifest)
        return _plan_from_manifest(manifest_path, manifest)
    except BaseException as primary:
        cleanup: list[BaseException] = []
        removed_public = False
        if published_manifest is not None:
            try:
                removed_public = (
                    unlink_file_if_identity(manifest_path, published_manifest)
                    or removed_public
                )
            except BaseException as error:
                cleanup.append(error)
        for path, identity in reversed(published):
            try:
                removed_public = unlink_file_if_identity(path, identity) or removed_public
            except BaseException as error:
                cleanup.append(error)
        if work.exists():
            try:
                shutil.rmtree(work)
            except BaseException as error:
                cleanup.append(error)
        if removed_public:
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


write_shards = write_parquet_shards
build_shards = write_parquet_shards
verify_shards = verify_shard_manifest


__all__ = [
    "COVERAGE_ALGORITHM",
    "SHARD_ALGORITHM",
    "SHARD_MANIFEST_SCHEMA",
    "ShardPlan",
    "build_shards",
    "load_shard_manifest",
    "verify_shard_manifest",
    "verify_shards",
    "write_parquet_shards",
    "write_shards",
]
