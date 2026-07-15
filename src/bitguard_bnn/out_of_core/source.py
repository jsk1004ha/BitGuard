from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from pandas.tseries.api import guess_datetime_format

from bitguard_bnn import data as data_module
from bitguard_bnn.config import resolve_path
from bitguard_bnn.constants import (
    META_COLUMNS,
    botiot_behavior,
    canonicalize_behavior,
    nbaiot_behavior,
    normalize_token,
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
    root: Path | None
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
    selected_rows: frozenset[int] | None
    schema: _Schema
    timestamp: _TimestampPlan | None


@dataclass(frozen=True, slots=True)
class _IterationPlan:
    source: _SourceSpec
    files: tuple[_FilePlan, ...]
    retained_uids: frozenset[str] | None
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


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _resolve_source(
    config: dict[str, Any], path_override: Path | None = None
) -> _SourceSpec:
    cfg = config["dataset"]
    kind = str(cfg["type"]).lower()
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
        root = Path(root)
        files = tuple(sorted(root.rglob("*.csv")))
        if not files:
            raise FileNotFoundError(f"no N-BaIoT CSV files under {root}")
        relative_paths = tuple(path.relative_to(root).as_posix() for path in files)
        return _SourceSpec(kind, files, relative_paths, root, label_overrides)
    pattern = path_override or cfg["path"]
    files = tuple(data_module._resolve_glob(config, pattern))
    if not files:
        if kind == "botiot":
            raise FileNotFoundError(f"no BoT-IoT CSV files match {pattern}")
        raise FileNotFoundError(f"no CSV files match {pattern}")
    project_root = Path(config.get("_project_root", "."))
    relative_paths = tuple(_relative_path(path, project_root) for path in files)
    return _SourceSpec(kind, files, relative_paths, None, label_overrides)


def _close_reader(reader: object) -> None:
    close = getattr(reader, "close", None)
    if callable(close):
        close()


def _scan_selected_rows(
    path: Path,
    *,
    chunk_size: int,
    max_rows: int | None,
    seed: int,
    budget: _SourceRowBudget,
) -> frozenset[int] | None:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive or null")
    rng = np.random.default_rng(seed)
    retained: list[tuple[float, int]] = []
    offset = 0
    reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
    try:
        for chunk in reader:
            rows = len(chunk)
            budget.consume(rows)
            if max_rows is not None:
                for source_index, key in zip(
                    range(offset, offset + rows), rng.random(rows)
                ):
                    entry = (-float(key), -source_index)
                    if len(retained) < max_rows:
                        heapq.heappush(retained, entry)
                    elif entry > retained[0]:
                        heapq.heapreplace(retained, entry)
            offset += rows
    finally:
        _close_reader(reader)
    if offset == 0:
        raise ValueError(f"empty CSV: {path}")
    if max_rows is None or offset <= max_rows:
        return None
    return frozenset(-source_index for _, source_index in retained)


def _iter_selected_raw(
    plan: _FilePlan | tuple[Path, frozenset[int] | None],
    chunk_size: int,
) -> Iterator[tuple[int, pd.DataFrame]]:
    if isinstance(plan, _FilePlan):
        path = plan.path
        selected_rows = plan.selected_rows
    else:
        path, selected_rows = plan
    offset = 0
    reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
    try:
        for chunk in reader:
            source_row_start = offset
            positions = np.arange(offset, offset + len(chunk), dtype=np.int64)
            offset += len(chunk)
            frame = chunk.copy()
            frame["__source_row_index"] = positions
            if selected_rows is not None:
                mask = np.fromiter(
                    (int(position) in selected_rows for position in positions),
                    dtype=bool,
                    count=len(positions),
                )
                frame = frame.loc[mask].copy()
            if not frame.empty:
                yield source_row_start, frame
    finally:
        _close_reader(reader)


def _schema_for(
    kind: str, frame: pd.DataFrame, cfg: dict[str, Any], path: Path
) -> _Schema:
    if kind == "nbaiot":
        return _Schema()
    if kind == "botiot":
        return _Schema(
            label=data_module._find_column(
                frame, cfg.get("label_column"), ["category", "label", "attack"]
            ),
            raw_attack=data_module._find_column(
                frame,
                cfg.get("raw_attack_column"),
                ["subcategory", "attack", "category"],
            ),
            device=data_module._find_column(
                frame, cfg.get("device_column"), ["saddr", "srcip", "device_id"]
            ),
            timestamp=data_module._find_column(
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
    path: Path,
    selected_rows: frozenset[int] | None,
    cfg: dict[str, Any],
    chunk_size: int,
) -> tuple[_Schema, _TimestampPlan | None]:
    schema: _Schema | None = None
    numeric_count = 0
    timestamp_count = 0
    first_timestamp: object | None = None
    for _, frame in _iter_selected_raw((path, selected_rows), chunk_size):
        if schema is None:
            schema = _schema_for(kind, frame, cfg, path)
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
    if schema is None:
        raise ValueError(f"empty CSV after sampling: {path}")
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
    assert source.root is not None
    relative = plan.path.relative_to(source.root)
    device = relative.parts[0] if len(relative.parts) > 1 else plan.path.parent.name
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
        return data_module._append_metadata(
            frame,
            dataset="nbaiot",
            source_file=plan.path,
            device_id=device,
            raw_attack=raw_attack,
            behavior_label=behavior,
            row_prefix="nbaiot",
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
        return data_module._append_metadata(
            features,
            dataset="botiot",
            source_file=plan.path,
            device_id=devices,
            raw_attack=raw_attack.map(normalize_token),
            behavior_label=behaviors,
            timestamp=timestamps,
            row_prefix="botiot",
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
    return data_module._append_metadata(
        features,
        dataset="csv",
        source_file=plan.path,
        device_id=devices,
        raw_attack=raw,
        behavior_label=labels,
        timestamp=timestamps,
        row_prefix="csv",
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
) -> frozenset[str] | None:
    if limit is None:
        return None
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
    return frozenset(entry[2] for heap in heaps.values() for entry in heap)


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
) -> _IterationPlan:
    source = _resolve_source(config, path_override)
    cfg = config["dataset"]
    chunk_size = int(cfg["chunk_size"])
    max_rows = cfg.get("max_rows_per_file") if apply_sampling_caps else None
    max_rows = int(max_rows) if max_rows is not None else None
    max_loaded_rows = cfg.get("max_loaded_rows") if apply_sampling_caps else None
    max_loaded_rows = int(max_loaded_rows) if max_loaded_rows is not None else None
    budget = _SourceRowBudget(max_loaded_rows)
    file_plans: list[_FilePlan] = []
    for path, relative_path in zip(source.files, source.relative_paths):
        selected_rows = _scan_selected_rows(
            path,
            chunk_size=chunk_size,
            max_rows=max_rows,
            seed=data_module._source_seed(int(config["experiment"]["seed"]), path),
            budget=budget,
        )
        schema, timestamp = _plan_schema_and_timestamp(
            source.kind, path, selected_rows, cfg, chunk_size
        )
        file_plans.append(
            _FilePlan(path, relative_path, selected_rows, schema, timestamp)
        )
    files = tuple(file_plans)
    class_limit = cfg.get("max_rows_per_class") if apply_sampling_caps else None
    class_limit = int(class_limit) if class_limit is not None else None
    retained_uids = _select_class_uids(
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
    return _IterationPlan(source, files, retained_uids, features)


def iter_normalized_chunks(
    config: dict[str, Any],
    path_override: Path | None = None,
    apply_sampling_caps: bool = True,
) -> Iterator[NormalizedChunk]:
    """Yield normalized dataset chunks without concatenating source frames."""

    plan = _build_iteration_plan(config, path_override, apply_sampling_caps)
    chunk_size = int(config["dataset"]["chunk_size"])
    for file_plan in plan.files:
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


def load_normalized_dataset(
    config: dict[str, Any], path_override: Path | None = None
) -> data_module.LoadedDataset:
    """Materialize normalized chunks at the legacy in-memory compatibility boundary."""

    chunks = list(iter_normalized_chunks(config, path_override))
    if not chunks:
        raise ValueError("dataset contains no rows")
    combined = (
        pd.concat([chunk.frame for chunk in chunks], ignore_index=True)
        .sample(frac=1, random_state=int(config["experiment"]["seed"]))
        .reset_index(drop=True)
    )
    cfg = config["dataset"]
    features = data_module._numeric_features(combined, cfg.get("drop_columns", []))
    source = _resolve_source(config, path_override)
    digests: dict[str, str] = {}
    for path, relative_path in zip(source.files, source.relative_paths):
        digest_key = (
            str(path.relative_to(source.root))
            if source.kind == "nbaiot" and source.root is not None
            else str(path)
        )
        digests[digest_key] = data_module._file_digest(path)
    provenance: dict[str, Any] = {
        "type": source.kind,
        "files": len(source.files),
        "sha256": digests,
        "has_wall_clock_time": (
            False
            if source.kind == "nbaiot"
            else bool(combined["timestamp"].notna().any())
        ),
    }
    if source.kind == "nbaiot":
        provenance.update(
            {
                "root": str(source.root),
                "notes": "N-BaIoT sequence_index is not a wall-clock timestamp.",
                "label_overrides": source.label_overrides,
            }
        )
    elif source.kind == "botiot":
        provenance["label_overrides"] = source.label_overrides
    return data_module.LoadedDataset(combined, features, provenance)
