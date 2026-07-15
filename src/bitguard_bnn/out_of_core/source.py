from __future__ import annotations

import heapq
import hashlib
import os
import stat
import warnings
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator

import numpy as np
import pandas as pd
from pandas.tseries.api import guess_datetime_format

from bitguard_bnn.config import resolve_path
from bitguard_bnn.constants import (
    META_COLUMNS,
    botiot_behavior,
    canonicalize_behavior,
    nbaiot_behavior,
    normalize_token,
)
from bitguard_bnn.out_of_core.common import (
    FileFingerprint,
    LoadedDataset,
    append_metadata,
    find_column,
    logical_source_id,
    normalize_logical_path,
    numeric_features,
    resolve_csv_files,
    source_sampling_key,
)


@dataclass(frozen=True, slots=True)
class NormalizedChunk:
    """One normalized source chunk with stable source coordinates."""

    frame: pd.DataFrame
    source_relative_path: str
    source_row_start: int


@dataclass(frozen=True, slots=True)
class _SourceSpec:
    kind: str
    files: tuple[Path, ...]
    relative_paths: tuple[str, ...]
    anchor: Path
    label_overrides: dict[str, str]


@dataclass(frozen=True, slots=True)
class _Schema:
    label: str | None = None
    raw_attack: str | None = None
    device: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class _TimestampPlan:
    numeric_mode: bool
    datetime_format: str | None


@dataclass(frozen=True, slots=True)
class _FilePlan:
    path: Path
    relative_path: str
    logical_identity: str
    fingerprint: FileFingerprint
    selected_rows: frozenset[int] | None
    schema: _Schema
    timestamp: _TimestampPlan | None


@dataclass(frozen=True, slots=True)
class _IterationPlan:
    source: _SourceSpec
    files: tuple[_FilePlan, ...]
    retained_uids: frozenset[str] | None
    materialization_order: tuple[str, ...] | None
    feature_columns: tuple[str, ...]


class _SourceRowBudget:
    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.consumed = 0
        if limit is not None and limit <= 0:
            raise ValueError("max_loaded_rows must be positive or null")

    def consume(self, rows: int) -> None:
        self.consumed += int(rows)
        if self.limit is not None and self.consumed > self.limit:
            raise MemoryError(
                "dataset.max_loaded_rows exceeded while reading CSV chunks; increase "
                "the explicit limit only after confirming available host memory"
            )


@dataclass(frozen=True, slots=True)
class _PinnedStat:
    device: int
    inode: int
    mode: int
    byte_size: int
    mtime_ns: int
    ctime_ns: int


class _HashingReader:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        block = self._handle.read(size)
        self._digest.update(block)
        self.bytes_read += len(block)
        return block

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def _pinned_stat(path: Path) -> _PinnedStat:
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"dataset source must be a regular file: {path}")
    return _PinnedStat(
        int(path_stat.st_dev),
        int(path_stat.st_ino),
        int(path_stat.st_mode),
        int(path_stat.st_size),
        int(path_stat.st_mtime_ns),
        int(path_stat.st_ctime_ns),
    )


def _assert_source_stat(
    path: Path,
    expected: _PinnedStat | FileFingerprint,
    actual: os.stat_result,
    phase: str,
) -> None:
    identity = (
        int(actual.st_dev),
        int(actual.st_ino),
        int(actual.st_mode),
        int(actual.st_size),
        int(actual.st_mtime_ns),
    )
    pinned = (
        expected.device,
        expected.inode,
        expected.mode,
        expected.byte_size,
        expected.mtime_ns,
    )
    if identity != pinned:
        raise RuntimeError(f"dataset source changed during {phase}: {path}")


class _VerifiedCsvPass:
    def __init__(
        self,
        path: Path,
        pinned: _PinnedStat | FileFingerprint,
        *,
        chunk_size: int,
        phase: str,
        expected_sha256: str | None,
    ) -> None:
        self.path = path
        self.pinned = pinned
        self.chunk_size = chunk_size
        self.phase = phase
        self.expected_sha256 = expected_sha256
        self.fingerprint: FileFingerprint | None = None

    def __iter__(self) -> Iterator[pd.DataFrame]:
        completed = False
        with self.path.open("rb", buffering=0) as handle:
            _assert_source_stat(self.path, self.pinned, os.fstat(handle.fileno()), self.phase)
            _assert_source_stat(self.path, self.pinned, self.path.lstat(), self.phase)
            hashing_reader = _HashingReader(handle)
            reader = pd.read_csv(
                hashing_reader,
                chunksize=self.chunk_size,
                low_memory=False,
            )
            try:
                for chunk in reader:
                    _assert_source_stat(
                        self.path,
                        self.pinned,
                        os.fstat(handle.fileno()),
                        self.phase,
                    )
                    _assert_source_stat(
                        self.path,
                        self.pinned,
                        self.path.lstat(),
                        self.phase,
                    )
                    yield chunk
                completed = True
            finally:
                _close_reader(reader)
                if completed:
                    _assert_source_stat(
                        self.path,
                        self.pinned,
                        os.fstat(handle.fileno()),
                        self.phase,
                    )
                    _assert_source_stat(
                        self.path,
                        self.pinned,
                        self.path.lstat(),
                        self.phase,
                    )
                    if hashing_reader.bytes_read != self.pinned.byte_size:
                        raise RuntimeError(
                            f"dataset source changed during {self.phase}: {self.path}"
                        )
                    digest = hashing_reader.sha256
                    if (
                        self.expected_sha256 is not None
                        and digest != self.expected_sha256
                    ):
                        raise RuntimeError(
                            f"dataset source changed during {self.phase}: {self.path}"
                        )
                    self.fingerprint = FileFingerprint(
                        self.pinned.device,
                        self.pinned.inode,
                        self.pinned.mode,
                        self.pinned.byte_size,
                        self.pinned.mtime_ns,
                        self.pinned.ctime_ns,
                        digest,
                    )


def _glob_anchor(path: Path) -> Path:
    text = str(path)
    wildcard_positions = [text.find(char) for char in "*?[" if char in text]
    if not wildcard_positions:
        return path if path.is_dir() else path.parent
    prefix = text[: min(wildcard_positions)]
    separator = max(prefix.rfind("/"), prefix.rfind("\\"))
    if separator < 0:
        return Path(path.anchor or ".")
    return Path(prefix[:separator] or path.anchor)


def _logical_paths(files: tuple[Path, ...], anchor: Path) -> tuple[str, ...]:
    relative_paths: list[str] = []
    seen: dict[str, str] = {}
    resolved_anchor = anchor.resolve()
    if resolved_anchor == Path(resolved_anchor.anchor):
        raise ValueError(f"dataset source anchor is too broad: {resolved_anchor}")
    for path in files:
        try:
            relative = path.resolve().relative_to(resolved_anchor).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"dataset source escapes its selected root {resolved_anchor}: {path}"
            ) from exc
        logical = normalize_logical_path(relative)
        duplicate_key = logical.casefold()
        if duplicate_key in seen:
            raise ValueError(
                "duplicate logical source path after normalization: "
                f"{seen[duplicate_key]!r} and {logical!r}"
            )
        seen[duplicate_key] = logical
        relative_paths.append(logical)
    return tuple(relative_paths)


def _resolve_source(
    config: dict[str, Any],
    path_override: Path | None = None,
    dataset_type: str | None = None,
) -> _SourceSpec:
    cfg = config["dataset"]
    kind = str(dataset_type or cfg["type"]).lower()
    if kind not in {"nbaiot", "botiot", "csv"}:
        raise ValueError(f"unsupported dataset.type: {kind}")
    label_overrides = (
        {
            normalize_token(key): canonicalize_behavior(value)
            for key, value in cfg.get("label_map", {}).items()
        }
        if kind != "csv"
        else {}
    )
    if kind == "nbaiot":
        root = path_override or resolve_path(config, cfg["path"])
        assert root is not None
        anchor = Path(root)
        files = tuple(sorted(anchor.rglob("*.csv")))
        if not files:
            raise FileNotFoundError(f"no N-BaIoT CSV files under {anchor}")
        relative_paths = _logical_paths(files, anchor)
        return _SourceSpec(kind, files, relative_paths, anchor, label_overrides)
    pattern = path_override or cfg["path"]
    files = resolve_csv_files(config, pattern)
    if not files:
        if kind == "botiot":
            raise FileNotFoundError(f"no BoT-IoT CSV files match {pattern}")
        raise FileNotFoundError(f"no CSV files match {pattern}")
    selected = resolve_path(config, pattern)
    assert selected is not None
    anchor = _glob_anchor(Path(selected))
    relative_paths = _logical_paths(files, anchor)
    return _SourceSpec(kind, files, relative_paths, anchor, label_overrides)


def _close_reader(reader: object) -> None:
    close = getattr(reader, "close", None)
    if callable(close):
        close()


def _scan_selected_rows(
    path: Path,
    relative_path: str,
    kind: str,
    pinned: _PinnedStat,
    *,
    chunk_size: int,
    max_rows: int | None,
    seed: int,
    budget: _SourceRowBudget,
) -> tuple[frozenset[int] | None, FileFingerprint, str]:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive or null")
    offset = 0
    csv_pass = _VerifiedCsvPass(
        path,
        pinned,
        chunk_size=chunk_size,
        phase="source planning",
        expected_sha256=None,
    )
    for chunk in csv_pass:
        rows = len(chunk)
        budget.consume(rows)
        offset += rows
    fingerprint = csv_pass.fingerprint
    assert fingerprint is not None
    identity = logical_source_id(kind, relative_path, fingerprint.sha256)
    if offset == 0:
        return frozenset(), fingerprint, identity
    if max_rows is None or offset <= max_rows:
        return None, fingerprint, identity
    retained: list[tuple[int, int]] = []
    for source_index in range(offset):
        key = source_sampling_key(seed, identity, source_index)
        entry = (-key, -source_index)
        if len(retained) < max_rows:
            heapq.heappush(retained, entry)
        elif entry > retained[0]:
            heapq.heapreplace(retained, entry)
    return (
        frozenset(-source_index for _, source_index in retained),
        fingerprint,
        identity,
    )


def _iter_selected_raw(
    plan: _FilePlan,
    chunk_size: int,
    *,
    include_empty: bool = False,
) -> Iterator[tuple[int, pd.DataFrame]]:
    offset = 0
    csv_pass = _VerifiedCsvPass(
        plan.path,
        plan.fingerprint,
        chunk_size=chunk_size,
        phase="CSV normalization pass",
        expected_sha256=plan.fingerprint.sha256,
    )
    for chunk in csv_pass:
        source_row_start = offset
        positions = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        frame = chunk.copy()
        frame["__source_row_index"] = positions
        if plan.selected_rows is not None:
            mask = np.fromiter(
                (int(position) in plan.selected_rows for position in positions),
                dtype=bool,
                count=len(positions),
            )
            frame = frame.loc[mask].copy()
        if include_empty or not frame.empty:
            yield source_row_start, frame


def _schema_for(
    kind: str, frame: pd.DataFrame, cfg: dict[str, Any], path: Path
) -> _Schema:
    if kind == "nbaiot":
        return _Schema()
    if kind == "botiot":
        return _Schema(
            label=find_column(
                frame, cfg.get("label_column"), ["category", "label", "attack"]
            ),
            raw_attack=find_column(
                frame,
                cfg.get("raw_attack_column"),
                ["subcategory", "attack", "category"],
            ),
            device=find_column(
                frame, cfg.get("device_column"), ["saddr", "srcip", "device_id"]
            ),
            timestamp=find_column(
                frame, cfg.get("time_column"), ["stime", "timestamp", "time"]
            ),
        )
    label = cfg.get("label_column", "behavior_label")
    if label not in frame:
        raise ValueError(f"label column {label!r} missing from {path}")
    raw = cfg.get("raw_attack_column", "raw_attack")
    device = cfg.get("device_column", "device_id")
    timestamp = cfg.get("time_column", "timestamp")
    return _Schema(
        label=label,
        raw_attack=raw if raw in frame else None,
        device=device if device in frame else None,
        timestamp=timestamp if timestamp in frame else None,
    )


def _plan_schema_and_timestamp(
    kind: str,
    plan: _FilePlan,
    cfg: dict[str, Any],
    chunk_size: int,
) -> tuple[_Schema, _TimestampPlan | None]:
    schema: _Schema | None = None
    numeric_count = 0
    timestamp_count = 0
    first_timestamp: object | None = None
    for _, frame in _iter_selected_raw(plan, chunk_size, include_empty=True):
        if schema is None:
            schema = _schema_for(kind, frame, cfg, plan.path)
        if frame.empty:
            continue
        if schema.timestamp is None:
            continue
        values = frame[schema.timestamp]
        converted = pd.to_numeric(values, errors="coerce")
        numeric_count += int(converted.notna().sum())
        timestamp_count += len(values)
        if first_timestamp is None:
            non_missing = values[values.notna()]
            if not non_missing.empty:
                first_timestamp = non_missing.iloc[0]
    assert schema is not None
    if schema.timestamp is None:
        return schema, None
    numeric_mode = timestamp_count > 0 and numeric_count / timestamp_count >= 0.95
    datetime_format: str | None = None
    if not numeric_mode and first_timestamp is not None:
        if isinstance(first_timestamp, str):
            datetime_format = guess_datetime_format(first_timestamp)
        if datetime_format is None:
            datetime_format = "mixed"
    return schema, _TimestampPlan(numeric_mode, datetime_format)


def _coerce_planned_timestamp(
    values: pd.Series, plan: _TimestampPlan
) -> pd.Series:
    if plan.numeric_mode:
        return pd.to_numeric(values, errors="coerce").astype(np.float64)
    options = (
        {"format": plan.datetime_format}
        if plan.datetime_format is not None
        else {}
    )
    parsed = pd.to_datetime(values, errors="coerce", utc=True, **options)
    seconds = pd.Series(np.nan, index=values.index, dtype=np.float64)
    valid = parsed.notna()
    seconds.loc[valid] = parsed.loc[valid].astype("int64") / 1_000_000_000.0
    return seconds


def _nbaiot_metadata(plan: _FilePlan, source: _SourceSpec) -> tuple[str, str, str]:
    relative = PurePosixPath(plan.relative_path)
    device = relative.parts[0] if len(relative.parts) > 1 else "source_root"
    stem = normalize_token(plan.path.stem)
    if "benign" in stem:
        raw_attack = "benign"
    else:
        family = normalize_token(plan.path.parent.name.replace("_attacks", ""))
        raw_attack = f"{family}_{stem}"
    behavior = source.label_overrides.get(raw_attack, nbaiot_behavior(raw_attack))
    return device, raw_attack, behavior


def _normalize_frame(
    frame: pd.DataFrame,
    plan: _FilePlan,
    source: _SourceSpec,
) -> pd.DataFrame:
    if source.kind == "nbaiot":
        device, raw_attack, behavior = _nbaiot_metadata(plan, source)
        return append_metadata(
            frame,
            dataset="nbaiot",
            logical_source=plan.relative_path,
            source_id=plan.logical_identity,
            device_id=device,
            raw_attack=raw_attack,
            behavior_label=behavior,
        )
    schema = plan.schema
    if source.kind == "botiot":
        category = (
            frame[schema.label]
            if schema.label is not None
            else pd.Series("unknown", index=frame.index)
        )
        raw_attack = frame[schema.raw_attack] if schema.raw_attack else category
        behaviors = [
            source.label_overrides.get(
                normalize_token(raw),
                source.label_overrides.get(
                    normalize_token(category_value),
                    botiot_behavior(category_value, raw),
                ),
            )
            for category_value, raw in zip(category, raw_attack)
        ]
        devices: str | pd.Series = (
            frame[schema.device].astype(str)
            if schema.device
            else f"source_{plan.path.stem}"
        )
        timestamps = (
            _coerce_planned_timestamp(frame[schema.timestamp], plan.timestamp)
            if schema.timestamp and plan.timestamp
            else None
        )
        metadata_sources = {
            schema.label,
            schema.raw_attack,
            schema.device,
            schema.timestamp,
        } - {None}
        features = frame.drop(columns=list(metadata_sources), errors="ignore")
        return append_metadata(
            features,
            dataset="botiot",
            logical_source=plan.relative_path,
            source_id=plan.logical_identity,
            device_id=devices,
            raw_attack=raw_attack.map(normalize_token),
            behavior_label=behaviors,
            timestamp=timestamps,
        )
    assert schema.label is not None
    labels = frame[schema.label].map(canonicalize_behavior)
    raw = (
        frame[schema.raw_attack].map(normalize_token)
        if schema.raw_attack
        else frame[schema.label].map(normalize_token)
    )
    devices = (
        frame[schema.device].astype(str)
        if schema.device
        else f"source_{plan.path.stem}"
    )
    timestamps = (
        _coerce_planned_timestamp(frame[schema.timestamp], plan.timestamp)
        if schema.timestamp and plan.timestamp
        else None
    )
    metadata_sources = {
        schema.label,
        schema.raw_attack,
        schema.device,
        schema.timestamp,
    } & set(frame.columns)
    features = frame.drop(columns=list(metadata_sources))
    return append_metadata(
        features,
        dataset="csv",
        logical_source=plan.relative_path,
        source_id=plan.logical_identity,
        device_id=devices,
        raw_attack=raw,
        behavior_label=labels,
        timestamp=timestamps,
    )


def _iter_file_normalized(
    plan: _FilePlan, source: _SourceSpec, chunk_size: int
) -> Iterator[NormalizedChunk]:
    for source_row_start, frame in _iter_selected_raw(plan, chunk_size):
        yield NormalizedChunk(
            _normalize_frame(frame, plan, source),
            plan.relative_path,
            source_row_start,
        )


def _select_class_uids(
    files: tuple[_FilePlan, ...],
    source: _SourceSpec,
    *,
    chunk_size: int,
    limit: int | None,
    seed: int,
) -> tuple[frozenset[str] | None, tuple[str, ...] | None]:
    if limit is None:
        return None, None
    if limit <= 0:
        raise ValueError("max_rows_per_class must be positive or null")
    rng = np.random.default_rng(seed)
    heaps: dict[str, list[tuple[float, int, str]]] = {}
    ordinals: dict[str, int] = {}
    for plan in files:
        label_order: list[str] = []
        seen: set[str] = set()
        for chunk in _iter_file_normalized(plan, source, chunk_size):
            for label in chunk.frame["behavior_label"].astype(str):
                if label not in seen:
                    seen.add(label)
                    label_order.append(label)
        for label in label_order:
            for chunk in _iter_file_normalized(plan, source, chunk_size):
                group = chunk.frame.loc[
                    chunk.frame["behavior_label"].astype(str).eq(label)
                ]
                if group.empty:
                    continue
                keys = rng.random(len(group))
                heap = heaps.setdefault(label, [])
                ordinal = ordinals.get(label, 0)
                for key, uid in zip(keys, group["row_uid"].astype(str)):
                    entry = (-float(key), -ordinal, uid)
                    ordinal += 1
                    if len(heap) < limit:
                        heapq.heappush(heap, entry)
                    elif entry > heap[0]:
                        heapq.heapreplace(heap, entry)
                ordinals[label] = ordinal
    ordered_uids: list[str] = []
    for label, heap in heaps.items():
        if ordinals[label] <= limit:
            retained = sorted(heap, key=lambda item: -item[1])
        else:
            retained = sorted(heap, key=lambda item: (-item[0], -item[1]))
        ordered_uids.extend(entry[2] for entry in retained)
    materialization_order = tuple(ordered_uids)
    return frozenset(materialization_order), materialization_order


def _filter_retained(
    frame: pd.DataFrame, retained_uids: frozenset[str] | None
) -> pd.DataFrame:
    if retained_uids is None:
        return frame
    return frame.loc[frame["row_uid"].isin(retained_uids)].copy()


def _plan_features(
    files: tuple[_FilePlan, ...],
    source: _SourceSpec,
    retained_uids: frozenset[str] | None,
    *,
    chunk_size: int,
    drop_columns: set[str],
) -> tuple[str, ...]:
    column_order: list[str] = []
    numeric: dict[str, bool] = {}
    row_count = 0
    excluded = META_COLUMNS | drop_columns
    for plan in files:
        for chunk in _iter_file_normalized(plan, source, chunk_size):
            frame = _filter_retained(chunk.frame, retained_uids)
            if frame.empty:
                continue
            row_count += len(frame)
            for column in frame.columns:
                if column not in column_order:
                    column_order.append(column)
                if column in excluded or numeric.get(column, False):
                    continue
                numeric[column] = bool(
                    pd.to_numeric(frame[column], errors="coerce").notna().any()
                )
    if row_count == 0:
        raise ValueError("dataset contains no rows")
    features = tuple(column for column in column_order if numeric.get(column, False))
    if not features:
        raise ValueError("no numeric feature columns were found")
    return features


def _build_iteration_plan(
    config: dict[str, Any],
    path_override: Path | None,
    apply_sampling_caps: bool,
    dataset_type: str | None = None,
) -> _IterationPlan:
    source = _resolve_source(config, path_override, dataset_type)
    cfg = config["dataset"]
    chunk_size = int(cfg["chunk_size"])
    max_rows = cfg.get("max_rows_per_file") if apply_sampling_caps else None
    max_rows = int(max_rows) if max_rows is not None else None
    max_loaded_rows = cfg.get("max_loaded_rows") if apply_sampling_caps else None
    max_loaded_rows = int(max_loaded_rows) if max_loaded_rows is not None else None
    budget = _SourceRowBudget(max_loaded_rows)
    class_limit = cfg.get("max_rows_per_class") if apply_sampling_caps else None
    class_limit = int(class_limit) if class_limit is not None else None
    file_plans: list[_FilePlan] = []
    has_rows = False
    for path, relative_path in zip(source.files, source.relative_paths):
        pinned = _pinned_stat(path)
        selected_rows, fingerprint, identity = _scan_selected_rows(
            path,
            relative_path,
            source.kind,
            pinned,
            chunk_size=chunk_size,
            max_rows=max_rows,
            seed=int(config["experiment"]["seed"]),
            budget=budget,
        )
        file_plan = _FilePlan(
            path,
            relative_path,
            identity,
            fingerprint,
            selected_rows,
            _Schema(),
            None,
        )
        schema, timestamp = _plan_schema_and_timestamp(
            source.kind, file_plan, cfg, chunk_size
        )
        file_plans.append(replace(file_plan, schema=schema, timestamp=timestamp))
        has_rows = has_rows or selected_rows != frozenset()
    files = tuple(file_plans)
    if not has_rows:
        if class_limit is not None:
            if class_limit <= 0:
                raise ValueError("max_rows_per_class must be positive or null")
            raise ValueError("dataset contains no rows")
        raise ValueError("no numeric feature columns were found")
    retained_uids, materialization_order = _select_class_uids(
        files,
        source,
        chunk_size=chunk_size,
        limit=class_limit,
        seed=int(config["experiment"]["seed"]),
    )
    features = _plan_features(
        files,
        source,
        retained_uids,
        chunk_size=chunk_size,
        drop_columns=set(cfg.get("drop_columns", [])),
    )
    return _IterationPlan(
        source,
        files,
        retained_uids,
        materialization_order,
        features,
    )


def _iter_planned_file_chunks(
    config: dict[str, Any], plan: _IterationPlan, file_plan: _FilePlan
) -> Iterator[NormalizedChunk]:
    chunk_size = int(config["dataset"]["chunk_size"])
    for chunk in _iter_file_normalized(file_plan, plan.source, chunk_size):
        frame = _filter_retained(chunk.frame, plan.retained_uids)
        if frame.empty:
            continue
        for column in plan.feature_columns:
            if column in frame:
                frame[column] = pd.to_numeric(
                    frame[column], errors="coerce"
                ).astype(np.float32)
        yield NormalizedChunk(
            frame=frame,
            source_relative_path=chunk.source_relative_path,
            source_row_start=chunk.source_row_start,
        )


def _iter_planned_chunks(
    config: dict[str, Any], plan: _IterationPlan
) -> Iterator[NormalizedChunk]:
    for file_plan in plan.files:
        yield from _iter_planned_file_chunks(config, plan, file_plan)


def iter_normalized_chunks(
    config: dict[str, Any],
    *,
    path_override: Path | None = None,
    apply_sampling_caps: bool = True,
) -> Iterator[NormalizedChunk]:
    """Yield normalized dataset chunks without concatenating source frames."""

    plan = _build_iteration_plan(config, path_override, apply_sampling_caps)
    yield from _iter_planned_chunks(config, plan)


def load_normalized_dataset(
    config: dict[str, Any],
    path_override: Path | None = None,
    *,
    dataset_type: str | None = None,
) -> LoadedDataset:
    """Materialize normalized chunks at the legacy in-memory compatibility boundary."""

    plan = _build_iteration_plan(config, path_override, True, dataset_type)
    frames: list[pd.DataFrame] = []
    for file_plan in plan.files:
        if (
            file_plan.selected_rows == frozenset()
            and plan.materialization_order is None
        ):
            header: pd.DataFrame | None = None
            header_pass = _VerifiedCsvPass(
                file_plan.path,
                file_plan.fingerprint,
                chunk_size=int(config["dataset"]["chunk_size"]),
                phase="header-only materialization",
                expected_sha256=file_plan.fingerprint.sha256,
            )
            for frame in header_pass:
                header = frame.iloc[0:0].copy()
            assert header is not None
            frames.append(_normalize_frame(header, file_plan, plan.source))
            continue
        frames.extend(
            chunk.frame
            for chunk in _iter_planned_file_chunks(config, plan, file_plan)
        )
    if not frames:
        raise ValueError("dataset contains no rows")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA",
            category=FutureWarning,
        )
        combined = pd.concat(frames, ignore_index=True)
    if plan.materialization_order is not None:
        order = {
            uid: position
            for position, uid in enumerate(plan.materialization_order)
        }
        combined["__materialization_order"] = combined["row_uid"].map(order)
        combined = combined.sort_values(
            "__materialization_order", kind="stable"
        ).drop(columns="__materialization_order")
    combined = combined.sample(
        frac=1, random_state=int(config["experiment"]["seed"])
    ).reset_index(drop=True)
    cfg = config["dataset"]
    features = numeric_features(combined, cfg.get("drop_columns", []))
    source = plan.source
    digests = {
        file_plan.relative_path: file_plan.fingerprint.sha256
        for file_plan in plan.files
    }
    provenance: dict[str, Any] = {
        "type": source.kind,
        "files": len(source.files),
        "sha256": digests,
        "source_identity": {
            "algorithm": "bitguard.logical-source.v1",
            "row_uid_algorithm": "bitguard.row-uid.v2",
            "sampling_algorithm": "bitguard.source-sampling.v1",
            "files": [
                {
                    "relative_path": file_plan.relative_path,
                    "byte_size": file_plan.fingerprint.byte_size,
                    "content_sha256": file_plan.fingerprint.sha256,
                    "logical_source_id": file_plan.logical_identity,
                }
                for file_plan in plan.files
            ],
        },
        "has_wall_clock_time": (
            False
            if source.kind == "nbaiot"
            else bool(combined["timestamp"].notna().any())
        ),
    }
    if source.kind == "nbaiot":
        provenance.update(
            {
                "root": str(source.anchor),
                "notes": "N-BaIoT sequence_index is not a wall-clock timestamp.",
                "label_overrides": source.label_overrides,
            }
        )
    elif source.kind == "botiot":
        provenance["label_overrides"] = source.label_overrides
    return LoadedDataset(combined, features, provenance)
