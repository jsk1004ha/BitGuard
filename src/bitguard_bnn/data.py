from __future__ import annotations

import glob
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import resolve_path
from .constants import (
    CANONICAL_LABELS,
    META_COLUMNS,
    botiot_behavior,
    canonicalize_behavior,
    nbaiot_behavior,
    normalize_token,
)


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    provenance: dict[str, Any]


@dataclass
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    manifest: dict[str, Any]


def _file_digest(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_chunks(
    path: Path,
    chunk_size: int,
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive or null")
    kept: pd.DataFrame | None = None
    rng = np.random.default_rng(seed)
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunk = chunk.copy()
        chunk["__source_row_index"] = chunk.index.to_numpy(dtype=np.int64)
        if max_rows is None:
            kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
            continue
        chunk["__sample_key"] = rng.random(len(chunk))
        kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        if len(kept) > max_rows * 2:
            kept = kept.nsmallest(max_rows, "__sample_key")
    if kept is None:
        raise ValueError(f"empty CSV: {path}")
    if max_rows is not None and len(kept) > max_rows:
        kept = kept.nsmallest(max_rows, "__sample_key")
    return (
        kept.drop(columns=["__sample_key"], errors="ignore")
        .sort_values("__source_row_index", kind="stable")
        .reset_index(drop=True)
    )


def _source_seed(seed: int, path: Path) -> int:
    token = int(hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:8], 16)
    return (seed + token) % (2**32 - 1)


def _append_metadata(
    frame: pd.DataFrame,
    *,
    dataset: str,
    source_file: Path,
    device_id: str | pd.Series,
    raw_attack: str | pd.Series,
    behavior_label: str | pd.Series,
    timestamp: pd.Series | None = None,
    row_prefix: str = "",
) -> pd.DataFrame:
    result = frame.copy()
    source = str(source_file.resolve())
    if "__source_row_index" in result:
        positions = result.pop("__source_row_index").to_numpy(dtype=np.int64)
    else:
        positions = np.arange(len(result), dtype=np.int64)
    result["dataset"] = dataset
    result["source_file"] = source
    result["sequence_index"] = positions
    result["device_id"] = device_id
    result["raw_attack"] = raw_attack
    result["behavior_label"] = behavior_label
    result["timestamp"] = timestamp if timestamp is not None else np.nan
    result["row_uid"] = [
        hashlib.sha256(f"{row_prefix}|{source}|{int(i)}".encode("utf-8")).hexdigest()
        for i in positions
    ]
    return result


class _FrameAccumulator:
    """Bound peak memory by retaining a priority reservoir per behavior class."""

    def __init__(self, per_class_limit: int | None, seed: int) -> None:
        self.limit = per_class_limit
        self.rng = np.random.default_rng(seed)
        self.frames: list[pd.DataFrame] = []
        self.groups: dict[str, pd.DataFrame] = {}

    def add(self, frame: pd.DataFrame) -> None:
        if self.limit is None:
            self.frames.append(frame)
            return
        if self.limit <= 0:
            raise ValueError("max_rows_per_class must be positive or null")
        for label, group in frame.groupby("behavior_label", sort=False):
            candidate = group.copy()
            candidate["__global_sample_key"] = self.rng.random(len(candidate))
            previous = self.groups.get(str(label))
            if previous is not None:
                candidate = pd.concat([previous, candidate], ignore_index=True)
            if len(candidate) > self.limit:
                candidate = candidate.nsmallest(self.limit, "__global_sample_key")
            self.groups[str(label)] = candidate

    def finish(self, seed: int) -> pd.DataFrame:
        pieces = self.frames if self.limit is None else list(self.groups.values())
        if not pieces:
            raise ValueError("dataset contains no rows")
        return (
            pd.concat(pieces, ignore_index=True)
            .drop(columns=["__global_sample_key"], errors="ignore")
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )


def _numeric_features(frame: pd.DataFrame, drop_columns: Iterable[str] = ()) -> list[str]:
    excluded = META_COLUMNS | set(drop_columns)
    candidates = [column for column in frame.columns if column not in excluded]
    numeric: list[str] = []
    for column in candidates:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            frame[column] = converted.astype(np.float32)
            numeric.append(column)
    if not numeric:
        raise ValueError("no numeric feature columns were found")
    return numeric


def load_nbaiot(config: dict[str, Any], path_override: Path | None = None) -> LoadedDataset:
    cfg = config["dataset"]
    root = path_override or resolve_path(config, cfg["path"])
    assert root is not None
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no N-BaIoT CSV files under {root}")
    seed = int(config["experiment"]["seed"])
    label_overrides = {
        normalize_token(key): canonicalize_behavior(value)
        for key, value in cfg.get("label_map", {}).items()
    }
    accumulator = _FrameAccumulator(cfg.get("max_rows_per_class"), seed)
    digests: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root)
        device = relative.parts[0] if len(relative.parts) > 1 else path.parent.name
        stem = normalize_token(path.stem)
        if "benign" in stem:
            raw_attack = "benign"
        else:
            family = normalize_token(path.parent.name.replace("_attacks", ""))
            raw_attack = f"{family}_{stem}"
        frame = _read_csv_chunks(
            path,
            int(cfg["chunk_size"]),
            cfg.get("max_rows_per_file"),
            _source_seed(int(config["experiment"]["seed"]), path),
        )
        behavior = label_overrides.get(raw_attack, nbaiot_behavior(raw_attack))
        frame = _append_metadata(
            frame,
            dataset="nbaiot",
            source_file=path,
            device_id=device,
            raw_attack=raw_attack,
            behavior_label=behavior,
            row_prefix="nbaiot",
        )
        accumulator.add(frame)
        digests[str(relative)] = _file_digest(path)
    combined = accumulator.finish(seed)
    features = _numeric_features(combined, cfg.get("drop_columns", []))
    return LoadedDataset(
        combined,
        features,
        {
            "type": "nbaiot",
            "root": str(root),
            "files": len(files),
            "sha256": digests,
            "has_wall_clock_time": False,
            "notes": "N-BaIoT sequence_index is not a wall-clock timestamp.",
            "label_overrides": label_overrides,
        },
    )


def _resolve_glob(config: dict[str, Any], pattern: str | Path) -> list[Path]:
    path = resolve_path(config, pattern)
    assert path is not None
    if any(char in str(path) for char in "*?["):
        return [Path(item) for item in sorted(glob.glob(str(path), recursive=True))]
    if path.is_dir():
        return sorted(path.rglob("*.csv"))
    return [path] if path.exists() else []


def load_botiot(config: dict[str, Any], path_override: Path | None = None) -> LoadedDataset:
    cfg = config["dataset"]
    files = _resolve_glob(config, path_override or cfg["path"])
    if not files:
        raise FileNotFoundError(f"no BoT-IoT CSV files match {path_override or cfg['path']}")
    seed = int(config["experiment"]["seed"])
    label_overrides = {
        normalize_token(key): canonicalize_behavior(value)
        for key, value in cfg.get("label_map", {}).items()
    }
    accumulator = _FrameAccumulator(cfg.get("max_rows_per_class"), seed)
    digests: dict[str, str] = {}
    for path in files:
        frame = _read_csv_chunks(
            path,
            int(cfg["chunk_size"]),
            cfg.get("max_rows_per_file"),
            _source_seed(int(config["experiment"]["seed"]), path),
        )
        label_col = _find_column(frame, cfg.get("label_column"), ["category", "label", "attack"])
        raw_col = _find_column(
            frame, cfg.get("raw_attack_column"), ["subcategory", "attack", "category"]
        )
        device_col = _find_column(frame, cfg.get("device_column"), ["saddr", "srcip", "device_id"])
        time_col = _find_column(frame, cfg.get("time_column"), ["stime", "timestamp", "time"])
        category = frame[label_col] if label_col else pd.Series("unknown", index=frame.index)
        raw_attack = frame[raw_col] if raw_col else category
        behaviors = [
            label_overrides.get(
                normalize_token(raw),
                label_overrides.get(normalize_token(cat), botiot_behavior(cat, raw)),
            )
            for cat, raw in zip(category, raw_attack)
        ]
        devices = frame[device_col].astype(str) if device_col else f"source_{path.stem}"
        timestamps = _coerce_timestamp(frame[time_col]) if time_col else None
        metadata_sources = {label_col, raw_col, device_col, time_col} - {None}
        frame = frame.drop(columns=list(metadata_sources), errors="ignore")
        frame = _append_metadata(
            frame,
            dataset="botiot",
            source_file=path,
            device_id=devices,
            raw_attack=raw_attack.map(normalize_token),
            behavior_label=behaviors,
            timestamp=timestamps,
            row_prefix="botiot",
        )
        accumulator.add(frame)
        digests[str(path)] = _file_digest(path)
    combined = accumulator.finish(seed)
    features = _numeric_features(combined, cfg.get("drop_columns", []))
    return LoadedDataset(
        combined,
        features,
        {
            "type": "botiot",
            "files": len(files),
            "sha256": digests,
            "has_wall_clock_time": bool(combined["timestamp"].notna().any()),
            "label_overrides": label_overrides,
        },
    )


def _find_column(frame: pd.DataFrame, preferred: str | None, candidates: list[str]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in [preferred, *candidates]:
        if candidate and candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _coerce_timestamp(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return numeric.astype(np.float64)
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    seconds = pd.Series(np.nan, index=values.index, dtype=np.float64)
    valid = parsed.notna()
    seconds.loc[valid] = parsed.loc[valid].astype("int64") / 1_000_000_000.0
    return seconds


def load_generic_csv(config: dict[str, Any], path_override: Path | None = None) -> LoadedDataset:
    cfg = config["dataset"]
    files = _resolve_glob(config, path_override or cfg["path"])
    if not files:
        raise FileNotFoundError(f"no CSV files match {path_override or cfg['path']}")
    seed = int(config["experiment"]["seed"])
    accumulator = _FrameAccumulator(cfg.get("max_rows_per_class"), seed)
    digests: dict[str, str] = {}
    for path in files:
        frame = _read_csv_chunks(
            path,
            int(cfg["chunk_size"]),
            cfg.get("max_rows_per_file"),
            _source_seed(int(config["experiment"]["seed"]), path),
        )
        label_col = cfg.get("label_column", "behavior_label")
        if label_col not in frame:
            raise ValueError(f"label column {label_col!r} missing from {path}")
        raw_col = cfg.get("raw_attack_column", "raw_attack")
        device_col = cfg.get("device_column", "device_id")
        time_col = cfg.get("time_column", "timestamp")
        labels = frame[label_col].map(canonicalize_behavior)
        raw = frame[raw_col].map(normalize_token) if raw_col in frame else frame[label_col].map(normalize_token)
        devices = frame[device_col].astype(str) if device_col in frame else f"source_{path.stem}"
        timestamps = _coerce_timestamp(frame[time_col]) if time_col in frame else None
        metadata_sources = {label_col, raw_col, device_col, time_col} & set(frame.columns)
        frame = frame.drop(columns=list(metadata_sources))
        frame = _append_metadata(
            frame,
            dataset="csv",
            source_file=path,
            device_id=devices,
            raw_attack=raw,
            behavior_label=labels,
            timestamp=timestamps,
            row_prefix="csv",
        )
        accumulator.add(frame)
        digests[str(path)] = _file_digest(path)
    combined = accumulator.finish(seed)
    features = _numeric_features(combined, cfg.get("drop_columns", []))
    return LoadedDataset(
        combined,
        features,
        {
            "type": "csv",
            "files": len(files),
            "sha256": digests,
            "has_wall_clock_time": bool(combined["timestamp"].notna().any()),
        },
    )


def load_dataset(config: dict[str, Any], path_override: Path | None = None) -> LoadedDataset:
    from .out_of_core.source import load_normalized_dataset

    return load_normalized_dataset(config, path_override)


def _split_train_validation(
    frame: pd.DataFrame,
    validation_share_of_subset: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = frame["behavior_label"] if frame["behavior_label"].value_counts().min() >= 2 else None
    train, validation = train_test_split(
        frame,
        test_size=validation_share_of_subset,
        random_state=seed,
        stratify=stratify,
    )
    return train.copy(), validation.copy()


def _split_random(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(cfg["seed"])
    stratify = frame["behavior_label"] if frame["behavior_label"].value_counts().min() >= 2 else None
    train_validation, test = train_test_split(
        frame,
        test_size=float(cfg["test_fraction"]),
        random_state=seed,
        stratify=stratify,
    )
    validation_share = float(cfg["validation_fraction"]) / (
        float(cfg["train_fraction"]) + float(cfg["validation_fraction"])
    )
    train, validation = _split_train_validation(train_validation, validation_share, seed)
    return train, validation, test.copy()


def _split_device(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    devices = [str(item) for item in cfg.get("held_out_devices", [])]
    if not devices:
        devices = [sorted(frame["device_id"].astype(str).unique())[-1]]
    mask = frame["device_id"].astype(str).isin(devices)
    if not mask.any() or mask.all():
        raise ValueError("device split must leave non-empty train and test sets")
    test = frame.loc[mask].copy()
    remainder = frame.loc[~mask].copy()
    validation_share = float(cfg["validation_fraction"]) / (
        float(cfg["train_fraction"]) + float(cfg["validation_fraction"])
    )
    train, validation = _split_train_validation(remainder, validation_share, int(cfg["seed"]))
    return train, validation, test


def _split_attack(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    held = {normalize_token(item) for item in cfg.get("held_out_attacks", [])}
    if not held:
        attack_options = sorted(set(frame.loc[frame["behavior_label"] != "benign", "raw_attack"]))
        if not attack_options:
            raise ValueError("attack split requires at least one attack subtype")
        held = {str(attack_options[-1])}
    mask = frame["raw_attack"].map(normalize_token).isin(held)
    if not mask.any() or mask.all():
        raise ValueError(f"held-out attacks {sorted(held)} do not produce valid train/test sets")
    unknown_test = frame.loc[mask].copy()
    unknown_test["behavior_label"] = "unknown_like"
    known = frame.loc[~mask].copy()
    train_validation, known_test = train_test_split(
        known,
        test_size=float(cfg["test_fraction"]),
        random_state=int(cfg["seed"]),
        stratify=known["behavior_label"] if known["behavior_label"].value_counts().min() >= 2 else None,
    )
    validation_share = float(cfg["validation_fraction"]) / (
        float(cfg["train_fraction"]) + float(cfg["validation_fraction"])
    )
    train, validation = _split_train_validation(
        train_validation, validation_share, int(cfg["seed"])
    )
    test = pd.concat([known_test, unknown_test], ignore_index=True)
    return train, validation, test


def _split_ordered(frame: pd.DataFrame, cfg: dict[str, Any], column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values([column, "source_file", "sequence_index"], kind="stable")
    n_train = int(len(ordered) * float(cfg["train_fraction"]))
    n_validation = int(len(ordered) * float(cfg["validation_fraction"]))
    train = ordered.iloc[:n_train].copy()
    validation = ordered.iloc[n_train : n_train + n_validation].copy()
    test = ordered.iloc[n_train + n_validation :].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("ordered split produced an empty partition")
    return train, validation, test


def _split_sequence_per_source(
    frame: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    partitions: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    for _, source in frame.groupby("source_file", sort=True):
        ordered = source.sort_values("sequence_index", kind="stable")
        n_train = int(len(ordered) * float(cfg["train_fraction"]))
        n_validation = int(len(ordered) * float(cfg["validation_fraction"]))
        partitions["train"].append(ordered.iloc[:n_train])
        partitions["validation"].append(ordered.iloc[n_train : n_train + n_validation])
        partitions["test"].append(ordered.iloc[n_train + n_validation :])
    result = tuple(pd.concat(partitions[name], ignore_index=True) for name in ("train", "validation", "test"))
    if min(map(len, result)) == 0:
        raise ValueError("sequence split produced an empty partition")
    return result  # type: ignore[return-value]


def _split_blocks(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    block_size = int(cfg.get("block_size", 10_000))
    if block_size <= 0:
        raise ValueError("split.block_size must be positive")
    groups = (
        frame["source_file"].astype(str)
        + "::"
        + (frame["sequence_index"].astype(np.int64) // block_size).astype(str)
    )
    outer = GroupShuffleSplit(
        n_splits=1, test_size=float(cfg["test_fraction"]), random_state=int(cfg["seed"])
    )
    train_validation_index, test_index = next(outer.split(frame, groups=groups))
    train_validation = frame.iloc[train_validation_index].copy()
    test = frame.iloc[test_index].copy()
    train_validation_groups = groups.iloc[train_validation_index]
    validation_share = float(cfg["validation_fraction"]) / (
        float(cfg["train_fraction"]) + float(cfg["validation_fraction"])
    )
    inner = GroupShuffleSplit(
        n_splits=1, test_size=validation_share, random_state=int(cfg["seed"]) + 1
    )
    train_index, validation_index = next(
        inner.split(train_validation, groups=train_validation_groups)
    )
    return (
        train_validation.iloc[train_index].copy(),
        train_validation.iloc[validation_index].copy(),
        test,
    )


def _split_manifest(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    strategy: str,
) -> dict[str, Any]:
    partitions = {"train": train, "validation": validation, "test": test}
    uid_sets = {name: set(part["row_uid"]) for name, part in partitions.items()}
    overlaps = {
        "train_validation": len(uid_sets["train"] & uid_sets["validation"]),
        "train_test": len(uid_sets["train"] & uid_sets["test"]),
        "validation_test": len(uid_sets["validation"] & uid_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"row leakage detected: {overlaps}")
    return {
        "strategy": strategy,
        "rows": {name: len(part) for name, part in partitions.items()},
        "class_counts": {
            name: part["behavior_label"].value_counts().sort_index().to_dict()
            for name, part in partitions.items()
        },
        "devices": {name: int(part["device_id"].nunique()) for name, part in partitions.items()},
        "row_uid_overlap": overlaps,
        "device_overlap": {
            "train_test": sorted(
                set(train["device_id"].astype(str)) & set(test["device_id"].astype(str))
            )[:100]
        },
        "unknown_in_train": int((train["behavior_label"] == "unknown_like").sum()),
    }


def make_split(dataset: LoadedDataset, config: dict[str, Any]) -> DataSplit:
    cfg = config["split"]
    strategy = str(cfg["strategy"])
    frame = dataset.frame
    if strategy == "random":
        train, validation, test = _split_random(frame, cfg)
    elif strategy == "device":
        train, validation, test = _split_device(frame, cfg)
    elif strategy == "attack":
        train, validation, test = _split_attack(frame, cfg)
    elif strategy == "time":
        if not dataset.provenance.get("has_wall_clock_time"):
            raise ValueError("time split requires a real timestamp; use sequence for N-BaIoT")
        if frame["timestamp"].isna().any():
            raise ValueError("time split does not accept missing timestamps")
        train, validation, test = _split_ordered(frame, cfg, "timestamp")
    elif strategy == "sequence":
        train, validation, test = _split_sequence_per_source(frame, cfg)
    elif strategy == "block":
        train, validation, test = _split_blocks(frame, cfg)
    elif strategy == "cross":
        raise ValueError("cross split is constructed by make_cross_split")
    else:
        raise ValueError(f"unsupported split strategy: {strategy}")
    manifest = _split_manifest(train, validation, test, strategy)
    manifest["provenance"] = dataset.provenance
    sampling_applied = any(
        config["dataset"].get(key) is not None
        for key in ("max_rows_per_file", "max_rows_per_class")
    )
    manifest["sampling_applied_before_split"] = sampling_applied
    manifest["natural_evaluation_distribution"] = not sampling_applied and strategy != "attack"
    manifest["temporal_continuity"] = strategy in {"time", "sequence", "device"} and not sampling_applied
    if not manifest["temporal_continuity"]:
        manifest["temporal_validity_note"] = (
            "State/action replay is exploratory: this split or pre-split sampling omits intervening events."
        )
    return DataSplit(train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), manifest)


def make_cross_split(
    source: LoadedDataset,
    target: LoadedDataset,
    config: dict[str, Any],
) -> tuple[DataSplit, list[str]]:
    shared = config["dataset"].get("shared_features")
    if not shared:
        raise ValueError(
            "cross-dataset evaluation requires dataset.shared_features; incompatible native schemas are never zero-padded"
        )
    source_schema = config["dataset"].get("feature_schema_id")
    target_schema = config["dataset"].get("cross_feature_schema_id")
    if not source_schema or not target_schema or source_schema != target_schema:
        raise ValueError(
            "cross-dataset evaluation requires equal feature_schema_id and cross_feature_schema_id "
            "to confirm matching definitions, units, and windows"
        )
    shared = [str(item) for item in shared]
    missing_source = sorted(set(shared) - set(source.feature_columns))
    missing_target = sorted(set(shared) - set(target.feature_columns))
    if missing_source or missing_target:
        raise ValueError(
            f"shared feature schema missing; source={missing_source}, target={missing_target}"
        )
    cfg = config["split"]
    validation_share = float(cfg["validation_fraction"]) / (
        float(cfg["train_fraction"]) + float(cfg["validation_fraction"])
    )
    train, validation = _split_train_validation(
        source.frame, validation_share, int(cfg["seed"])
    )
    test = target.frame.copy()
    manifest = _split_manifest(train, validation, test, "cross")
    manifest["source_provenance"] = source.provenance
    manifest["target_provenance"] = target.provenance
    manifest["shared_features"] = shared
    manifest["feature_schema_id"] = source_schema
    sampling_applied = any(
        config["dataset"].get(key) is not None
        for key in ("max_rows_per_file", "max_rows_per_class")
    )
    manifest["sampling_applied_before_split"] = sampling_applied
    manifest["natural_evaluation_distribution"] = not sampling_applied
    manifest["temporal_continuity"] = not sampling_applied
    return DataSplit(train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), manifest), shared


def validate_labels(frame: pd.DataFrame) -> None:
    invalid = sorted(set(frame["behavior_label"]) - set(CANONICAL_LABELS))
    if invalid:
        raise ValueError(f"invalid canonical labels: {invalid}")
