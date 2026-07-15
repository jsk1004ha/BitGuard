from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn


SPLIT_MANIFEST_SCHEMA = "bitguard.split-manifest.v1"
SPLIT_MANIFEST_SEMANTICS = "bitguard.split-manifest-semantics.v1"

_SEMANTIC_TOP_LEVEL_FIELDS = (
    "schema_version",
    "fingerprint",
    "strategy",
    "counts",
    "class_counts",
    "source_coverage",
    "source_manifest_fingerprint",
    "declared_source_manifest_fingerprint",
    "config_signature",
    "checks",
    "rejections",
    "inspection",
    "schema",
    "schemas",
    "schema_fingerprint",
    "algorithm_versions",
    "held_out",
    "ordering_boundaries",
)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int


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


def split_manifest_semantics(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        semantic = {name: payload[name] for name in _SEMANTIC_TOP_LEVEL_FIELDS}
        membership = payload["membership"]
        semantic["membership"] = {
            name: membership[name]
            for name in ("path", "logical_digest", "rows", "uid_sorted")
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("split manifest is missing semantic fields") from exc
    return {
        "algorithm": SPLIT_MANIFEST_SEMANTICS,
        "manifest": semantic,
    }


def split_manifest_semantic_fingerprint(payload: Mapping[str, Any]) -> str:
    return stable_fingerprint(split_manifest_semantics(payload))


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


def file_identity(path: Path) -> FileIdentity:
    status = path.stat(follow_symlinks=False)
    return FileIdentity(
        device=int(status.st_dev),
        inode=int(status.st_ino),
        mode=int(status.st_mode),
        size=int(status.st_size),
        mtime_ns=int(status.st_mtime_ns),
    )


def unlink_file_if_identity(path: Path, identity: FileIdentity) -> bool:
    if not path.exists():
        return False
    if file_identity(path) != identity:
        raise RuntimeError(f"published artifact identity changed during rollback: {path}")
    path.unlink()
    return True


def attach_cleanup_context(primary: BaseException, message: str) -> None:
    """Preserve cleanup diagnostics on Python 3.10 and newer runtimes."""

    try:
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            add_note(message)
            return
        existing = getattr(primary, "__bitguard_cleanup_notes__", ())
        if not isinstance(existing, tuple):
            existing = ()
        setattr(primary, "__bitguard_cleanup_notes__", (*existing, message))
    except Exception:
        return


def _raise_with_cleanup(
    primary: BaseException, cleanup: list[BaseException]
) -> NoReturn:
    if cleanup:
        for error in cleanup:
            attach_cleanup_context(
                primary,
                f"cleanup failure: {type(error).__name__}: {error}",
            )
        raise primary from cleanup[0]
    raise primary


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> FileIdentity:
    """Publish new JSON durably and roll back only the exact file on failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    published: FileIdentity | None = None
    renamed = False
    try:
        with partial.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to replace existing manifest: {path}")
        os.replace(partial, path)
        renamed = True
        published = file_identity(path)
        _fsync_directory(path.parent)
        return published
    except BaseException as primary:
        cleanup: list[BaseException] = []
        removed_directory_entry = False
        try:
            if published is not None:
                removed_directory_entry = unlink_file_if_identity(path, published)
            elif renamed and path.exists():
                if path.read_bytes() != encoded:
                    raise RuntimeError(
                        f"published manifest identity changed during rollback: {path}"
                    )
                path.unlink()
                removed_directory_entry = True
        except BaseException as error:
            cleanup.append(error)
        try:
            partial_existed = partial.exists()
            partial.unlink(missing_ok=True)
            removed_directory_entry = removed_directory_entry or partial_existed
        except BaseException as error:
            cleanup.append(error)
        if removed_directory_entry:
            try:
                _fsync_directory(path.parent)
            except BaseException as error:
                cleanup.append(error)
        _raise_with_cleanup(primary, cleanup)


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
    "SPLIT_MANIFEST_SEMANTICS",
    "FileIdentity",
    "SourceRowRecord",
    "SplitPlan",
    "attach_cleanup_context",
    "canonical_json_bytes",
    "file_identity",
    "manifest_path_for_membership",
    "read_split_manifest",
    "split_manifest_semantic_fingerprint",
    "split_manifest_semantics",
    "stable_fingerprint",
    "unlink_file_if_identity",
]
