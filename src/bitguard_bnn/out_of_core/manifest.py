from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SPLIT_MANIFEST_SCHEMA = "bitguard.split-manifest.v1"


@dataclass(frozen=True, slots=True)
class SourceRowRecord:
    row_uid: str
    source_file: str
    source_row: int
    behavior_label: str
    raw_attack: str
    device_id: str
    timestamp: float | None


@dataclass(frozen=True, slots=True)
class SplitPlan:
    strategy: str
    train_count: int
    validation_count: int
    test_count: int
    membership_path: Path
    fingerprint: str


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically without accepting non-finite numbers."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def stable_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def manifest_path_for_membership(membership_path: Path | str) -> Path:
    return Path(membership_path).with_suffix(".manifest.json")


def read_split_manifest(plan_or_path: SplitPlan | Path | str) -> dict[str, Any]:
    path = (
        manifest_path_for_membership(plan_or_path.membership_path)
        if isinstance(plan_or_path, SplitPlan)
        else Path(plan_or_path)
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported split manifest schema: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace a JSON file; caller owns rollback of related artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("wb") as handle:
            handle.write(canonical_json_bytes(dict(payload)))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    finally:
        partial.unlink(missing_ok=True)


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
        # Windows does not expose a portable directory flush through os.fsync.
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


__all__ = [
    "SPLIT_MANIFEST_SCHEMA",
    "SourceRowRecord",
    "SplitPlan",
    "canonical_json_bytes",
    "manifest_path_for_membership",
    "read_split_manifest",
    "stable_fingerprint",
]
