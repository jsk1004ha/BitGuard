from __future__ import annotations

import glob
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd

from bitguard_bnn.config import resolve_path
from bitguard_bnn.constants import META_COLUMNS


LOGICAL_SOURCE_ALGORITHM = "bitguard.logical-source.v1"
ROW_UID_ALGORITHM = "bitguard.row-uid.v2"
SOURCE_SAMPLING_ALGORITHM = "bitguard.source-sampling.v1"


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    device: int
    inode: int
    mode: int
    byte_size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


def _canonical_digest(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def normalize_logical_path(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\\", "/"))
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"invalid logical source path: {value!r}")
    return path.as_posix()


def logical_source_id(kind: str, relative_path: str, content_sha256: str) -> str:
    return _canonical_digest(
        {
            "algorithm": LOGICAL_SOURCE_ALGORITHM,
            "content_sha256": content_sha256,
            "dataset_type": str(kind).lower(),
            "relative_path": normalize_logical_path(relative_path),
        }
    ).hex()


def source_sampling_seed(seed: int, source_id: str) -> int:
    digest = _canonical_digest(
        {
            "algorithm": SOURCE_SAMPLING_ALGORITHM,
            "experiment_seed": int(seed),
            "source_id": source_id,
        }
    )
    return int.from_bytes(digest[:4], "big", signed=False)


def source_sampling_key(seed: int, source_id: str, row_index: int) -> int:
    return int.from_bytes(
        _canonical_digest(
            {
                "algorithm": SOURCE_SAMPLING_ALGORITHM,
                "experiment_seed": int(seed),
                "row_index": int(row_index),
                "source_id": source_id,
            }
        )[:8],
        "big",
        signed=False,
    )


def row_uid(source_id: str, row_index: int) -> str:
    return _canonical_digest(
        {
            "algorithm": ROW_UID_ALGORITHM,
            "row_index": int(row_index),
            "source_id": source_id,
        }
    ).hex()


def append_metadata(
    frame: pd.DataFrame,
    *,
    dataset: str,
    logical_source: str,
    source_id: str,
    device_id: str | pd.Series,
    raw_attack: str | pd.Series,
    behavior_label: str | pd.Series,
    timestamp: pd.Series | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if "__source_row_index" in result:
        positions = result.pop("__source_row_index").to_numpy(dtype=np.int64)
    else:
        positions = np.arange(len(result), dtype=np.int64)
    result["dataset"] = dataset
    result["source_file"] = normalize_logical_path(logical_source)
    result["sequence_index"] = positions
    result["device_id"] = device_id
    result["raw_attack"] = raw_attack
    result["behavior_label"] = behavior_label
    result["timestamp"] = timestamp if timestamp is not None else np.nan
    result["row_uid"] = [row_uid(source_id, int(position)) for position in positions]
    return result


def numeric_features(
    frame: pd.DataFrame, drop_columns: Iterable[str] = ()
) -> list[str]:
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


def find_column(
    frame: pd.DataFrame, preferred: str | None, candidates: list[str]
) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in [preferred, *candidates]:
        if candidate and candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def resolve_csv_files(config: dict[str, Any], pattern: str | Path) -> tuple[Path, ...]:
    path = resolve_path(config, pattern)
    assert path is not None
    if any(char in str(path) for char in "*?["):
        return tuple(Path(item) for item in sorted(glob.glob(str(path), recursive=True)))
    if path.is_dir():
        return tuple(sorted(path.rglob("*.csv")))
    return (path,) if path.exists() else ()
