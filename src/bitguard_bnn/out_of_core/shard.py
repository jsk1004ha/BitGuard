from __future__ import annotations

import hashlib
import heapq
import ctypes
import errno
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    BinaryIO,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    NoReturn,
    Sequence,
    cast,
)

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
)
from bitguard_bnn.out_of_core.source import NormalizedChunk

SHARD_MANIFEST_SCHEMA = "bitguard.shard-manifest.v2"
SHARD_ALGORITHM = "bitguard.immutable-parquet-shards.v2"
COVERAGE_ALGORITHM = "bitguard.external-uid-coverage.v1"
_SHARD_BUILD_TRANSACTION_SCHEMA = "bitguard.shard-build-transaction.v1"
_SHARD_LOCK_SCHEMA = "bitguard.shard-build-lock.v1"
_SHARD_LOCK_WAL_MAGIC = b"BGLOCK2\0"
_SHARD_LOCK_WAL_HEADER = struct.Struct(">8sQQ32s")
_SHARD_LOCK_WAL_SLOT_CAPACITY = 8 * 1024 * 1024
_SHARD_LOCK_WAL_SLOT_COUNT = 2
_SHARD_PRIVATE_TRASH = ".bitguard-private-trash"
_SHARD_PRIVATE_TRASH_MAX_ENTRIES = 16
_SHARD_VERIFICATION_TRASH = ".bitguard-verification-trash"
_SHARD_WORK_OWNER_FILE = ".bitguard-shard-owner.json"
_SHARD_WORK_OWNER_INITIALIZING_FILE = ".bitguard-shard-owner.initializing"

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
    merge_read_rows: int
    max_run_rows: int = 0
    run_count: int = 0
    max_merge_fan_in_observed: int = 0
    temporary_bytes_peak: int = 0
    merge_input_rows_buffered: int = 0
    max_merge_input_rows_buffered: int = 0

    def record_run(self, rows: int) -> None:
        self.max_run_rows = max(self.max_run_rows, int(rows))
        self.run_count += 1
        self.observe_disk()

    def record_merge(self, fan_in: int) -> None:
        self.max_merge_fan_in_observed = max(
            self.max_merge_fan_in_observed, int(fan_in)
        )

    def open_merge_batch(self, rows: int) -> None:
        if rows < 0 or rows > self.merge_read_rows:
            raise RuntimeError("merge input batch exceeded configured row bound")
        self.merge_input_rows_buffered += rows
        self.max_merge_input_rows_buffered = max(
            self.max_merge_input_rows_buffered,
            self.merge_input_rows_buffered,
        )

    def close_merge_batch(self, rows: int) -> None:
        self.merge_input_rows_buffered -= rows
        if self.merge_input_rows_buffered < 0:
            raise RuntimeError("merge input row accounting underflow")

    def observe_disk(self) -> None:
        total = 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        self.temporary_bytes_peak = max(self.temporary_bytes_peak, total)


def _write_all(handle: BinaryIO, payload: bytes | bytearray | memoryview) -> None:
    """Write every byte or fail without silently accepting a short write."""

    remaining = memoryview(payload)
    while remaining:
        written = handle.write(remaining)
        if written is None:
            raise OSError("file write did not report progress")
        if written <= 0:
            raise OSError("file write made no forward progress")
        remaining = remaining[written:]


def _write_all_descriptor(descriptor: int, payload: bytes | memoryview) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("descriptor write made no forward progress")
        remaining = remaining[written:]


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


def _publish_file_no_replace(
    source: Path,
    destination: Path,
    *,
    publication_index: int | None = None,
) -> FileIdentity:
    """Atomically publish a same-volume file without an overwrite race."""

    expected = file_identity(source)
    _move_entry_to_quarantine_exact(source, destination, expected, kind="file")
    actual = file_identity(destination)
    if actual != expected:
        raise RuntimeError(f"published artifact identity changed: {destination}")
    if publication_index is not None:
        _after_public_shard_link_boundary(publication_index, source, destination)
        _after_public_shard_source_removal(publication_index, source, destination)
    return actual


def _after_manifest_temporary_intent_boundary(_partial: Path) -> None:
    """Test seam after manifest temporary creation intent is durable."""


def _after_manifest_temporary_created_anchor_boundary(_partial: Path) -> None:
    """Test seam after the empty manifest temporary inode is anchored."""


def _after_manifest_temporary_payload_boundary(_partial: Path) -> None:
    """Test seam after manifest bytes and their full identity are anchored."""


def _after_manifest_publish_link_boundary(_partial: Path, _path: Path) -> None:
    """Test seam after manifest no-replace link and before private-name removal."""


def _write_manifest_no_replace(
    path: Path, payload: Mapping[str, Any], transaction_lock: BinaryIO
) -> FileIdentity:
    partial = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    fingerprint = stable_fingerprint(dict(payload))
    record: dict[str, Any] = {
        "phase": "initializing",
        "path": str(partial),
        "destination": str(path),
        "fingerprint": fingerprint,
    }
    anchor = _read_lock_anchor(transaction_lock)
    if anchor.get("manifest") is not None:
        raise RuntimeError("owned manifest temporary is still active")
    anchor["manifest"] = record
    _write_lock_anchor(transaction_lock, anchor)
    _after_manifest_temporary_intent_boundary(partial)
    expected: FileIdentity | None = None
    published: FileIdentity | None = None
    try:
        with partial.open("xb") as handle:
            record = {
                **record,
                "phase": "created",
                "instance": _status_instance(os.fstat(handle.fileno())),
            }
            anchor = _read_lock_anchor(transaction_lock)
            anchor["manifest"] = record
            _write_lock_anchor(transaction_lock, anchor)
            _after_manifest_temporary_created_anchor_boundary(partial)
            _write_all(handle, canonical_json_bytes(dict(payload)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(partial.parent)
        expected = file_identity(partial)
        record = {
            **record,
            "phase": "active",
            "identity": _identity_payload(expected),
        }
        anchor = _read_lock_anchor(transaction_lock)
        anchor["manifest"] = record
        _write_lock_anchor(transaction_lock, anchor)
        _after_manifest_temporary_payload_boundary(partial)
        _move_entry_to_quarantine_exact(partial, path, expected, kind="file")
        _after_manifest_publish_link_boundary(partial, path)
        if _path_entry_exists(partial) or file_identity(path) != expected:
            raise RuntimeError(f"published artifact identity changed: {path}")
        published = file_identity(path)
        anchor = _read_lock_anchor(transaction_lock)
        anchor["manifest"] = {
            **record,
            "phase": "published",
            "identity": _identity_payload(published),
        }
        _write_lock_anchor(transaction_lock, anchor)
        return published
    except BaseException:
        # The lock-anchor WAL owns recovery.  Immediate pathname cleanup could
        # target a replacement installed after the publication failure.
        raise


def _path_entry_exists(path: Path) -> bool:
    """Return whether a directory entry exists, including dangling links."""

    return os.path.lexists(path)


def _entry_instance(path: Path) -> list[int] | None:
    try:
        status = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    return [int(status.st_dev), int(status.st_ino), int(status.st_mode)]


def _file_identity_payload(path: Path) -> list[int]:
    identity = file_identity(path)
    return [
        identity.device,
        identity.inode,
        identity.mode,
        identity.size,
        identity.mtime_ns,
    ]


def _identity_payload_matches(path: Path, expected: object) -> bool:
    if (
        not isinstance(expected, list)
        or len(expected) != 5
        or any(type(value) is not int for value in expected)
        or not _path_entry_exists(path)
    ):
        return False
    try:
        return _file_identity_payload(path) == expected
    except OSError:
        return False


def _before_identity_bound_entry_removal(_path: Path, _kind: str) -> None:
    """Test seam after an entry is pinned and immediately before removal."""


def _removal_status_matches(
    status: os.stat_result,
    expected: FileIdentity | Sequence[int],
    *,
    kind: str,
) -> bool:
    if kind == "file":
        if not isinstance(expected, FileIdentity) or not stat.S_ISREG(status.st_mode):
            return False
        return (
            int(status.st_dev),
            int(status.st_ino),
            int(status.st_mode),
            int(status.st_size),
            int(status.st_mtime_ns),
        ) == (
            expected.device,
            expected.inode,
            expected.mode,
            expected.size,
            expected.mtime_ns,
        )
    directory_expected = cast(Sequence[int], expected)
    return (
        kind == "directory"
        and len(directory_expected) == 3
        and stat.S_ISDIR(status.st_mode)
        and [int(status.st_dev), int(status.st_ino), int(status.st_mode)]
        == list(directory_expected)
    )


def _entry_matches_expected(
    path: Path,
    expected: FileIdentity | Sequence[int],
    *,
    kind: str,
) -> bool:
    if not _path_entry_exists(path) or _path_is_link_like(path):
        return False
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return _removal_status_matches(status, expected, kind=kind)


def _move_entry_to_quarantine_exact(
    source: Path,
    quarantine: Path,
    expected: FileIdentity | Sequence[int],
    *,
    kind: str,
) -> None:
    """Move only ``expected`` to a no-clobber name, preserving any replacement.

    The durable caller records ``quarantine`` before invoking this operation.  If
    the source is exchanged at the last syscall boundary, the moved foreign entry
    is restored with another no-replace rename whenever its original name is free.
    A replacement that appears after the owned object moved is never touched.
    """

    if kind not in {"file", "directory"}:
        raise ValueError(f"unsupported quarantine entry kind: {kind}")
    if not _entry_matches_expected(source, expected, kind=kind):
        raise RuntimeError(f"identity-bound {kind} changed before quarantine: {source}")
    if _path_entry_exists(quarantine):
        raise RuntimeError(f"quarantine entry already exists: {quarantine}")
    _before_identity_bound_entry_removal(source, kind)
    _rename_directory_no_replace(source, quarantine)
    _fsync_directory(source.parent)
    if quarantine.parent != source.parent:
        _fsync_directory(quarantine.parent)
    if _entry_matches_expected(quarantine, expected, kind=kind):
        if _path_entry_exists(source):
            raise RuntimeError(f"foreign {kind} replaced quarantined entry: {source}")
        return

    # The source was replaced after validation but before the rename syscall.
    # Restore that foreign object without overwriting a second late replacement.
    if _path_entry_exists(quarantine) and not _path_entry_exists(source):
        try:
            _rename_directory_no_replace(quarantine, source)
            _fsync_directory(source.parent)
            if quarantine.parent != source.parent:
                _fsync_directory(quarantine.parent)
        except BaseException:
            pass
    raise RuntimeError(f"identity-bound {kind} changed during quarantine: {source}")


def _remove_entry_by_windows_handle(
    path: Path,
    expected: FileIdentity | Sequence[int],
    *,
    kind: str,
) -> None:
    """Delete the opened Windows file object, never a later pathname occupant."""

    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int

    delete_access = 0x00010000
    share_read_write_delete = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    open_reparse_point = 0x00200000
    backup_semantics = 0x02000000 if kind == "directory" else 0
    raw_handle = create_file(
        str(path),
        delete_access,
        share_read_write_delete,
        None,
        open_existing,
        open_reparse_point | backup_semantics,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if raw_handle in {None, invalid_handle}:
        raise ctypes.WinError(ctypes.get_last_error())

    descriptor = -1
    try:
        descriptor = msvcrt.open_osfhandle(int(raw_handle), os.O_RDONLY)
        raw_handle = None
        if not _removal_status_matches(os.fstat(descriptor), expected, kind=kind):
            raise RuntimeError(f"identity-bound {kind} changed before removal: {path}")
        _before_identity_bound_entry_removal(path, kind)

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("delete_file", ctypes.c_ubyte)]

        disposition = FileDispositionInfo(1)
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(descriptor))
        if not set_information(
            handle,
            4,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        elif raw_handle not in {None, invalid_handle}:
            kernel32.CloseHandle(ctypes.c_void_p(raw_handle))


def _remove_entry_if_identity(
    path: Path,
    expected: FileIdentity | Sequence[int],
    *,
    kind: str,
) -> bool:
    """Remove only the entry object represented by ``expected``.

    Windows pins the object with a DELETE handle. Portable callers must quarantine
    and retain the object instead of falling back to a pathname deletion.
    """

    if not _path_entry_exists(path):
        return False
    if os.name != "nt":
        raise RuntimeError("identity-bound pathname removal is unsupported")
    _remove_entry_by_windows_handle(path, expected, kind=kind)
    if _path_entry_exists(path):
        raise RuntimeError(f"foreign {kind} replaced removed entry: {path}")
    return True


def _after_private_trash_entry_removal_boundary(_path: Path) -> None:
    """Test seam after unlink+rmdir fsync and before WAL progress advances."""


def _delete_from_private_trash(
    path: Path,
    expected: FileIdentity | Sequence[int],
    *,
    kind: str,
    transaction_lock: BinaryIO,
    allow_truncated_file: bool = False,
) -> None:
    """Delete inside the anchored 0700 directory via its exact directory inode.

    The same operating-system user is part of the trust boundary; group and
    other users of a shared output root cannot traverse the private directory.
    The bounded WAL names every in-flight entry before it reaches this function.
    """

    anchor = _read_lock_anchor(transaction_lock, allow_unrecorded_retirement=True)
    output = Path(str(anchor["lock_path"])).parent
    trash = _validate_private_trash_anchor(output, anchor.get("private_trash"))
    if trash is None or path.parent != trash:
        raise RuntimeError(f"unsafe shard private trash path: {path}")
    if os.name == "nt":
        # Platform-neutral tests exercise the private-trash state machine on
        # Windows, where directory descriptors are unavailable.  Production
        # Windows retirement uses identity-bound kernel handles instead.
        before = path.stat(follow_symlinks=False)
        if not _removal_status_matches(before, expected, kind=kind):
            raise RuntimeError(f"retired shard {kind} identity changed: {path}")
        _before_identity_bound_entry_removal(path, kind)
        if not _removal_status_matches(
            path.stat(follow_symlinks=False), expected, kind=kind
        ):
            raise RuntimeError(f"retired shard {kind} identity changed: {path}")
        path.unlink() if kind == "file" else path.rmdir()
        _fsync_directory(trash)
        _after_private_trash_entry_removal_boundary(path)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(trash, flags)
    try:
        trash_status = os.fstat(descriptor)
        trash_record = cast(dict[str, Any], anchor["private_trash"])
        if _status_instance(trash_status) != trash_record.get("identity"):
            raise RuntimeError("shard private trash identity changed")

        def entry_status() -> os.stat_result:
            if os.name == "nt":
                return path.stat(follow_symlinks=False)
            return os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)

        before = entry_status()
        matches = _removal_status_matches(before, expected, kind=kind)
        if (
            not matches
            and kind == "file"
            and allow_truncated_file
            and isinstance(expected, FileIdentity)
        ):
            matches = (
                stat.S_ISREG(before.st_mode)
                and (int(before.st_dev), int(before.st_ino), int(before.st_mode))
                == (expected.device, expected.inode, expected.mode)
                and int(before.st_nlink) == 1
                and int(before.st_size) in {0, expected.size}
            )
        if not matches:
            raise RuntimeError(f"retired shard {kind} identity changed: {path}")
        _before_identity_bound_entry_removal(path, kind)
        if not _removal_status_matches(entry_status(), expected, kind=kind):
            if not (
                kind == "file"
                and allow_truncated_file
                and isinstance(expected, FileIdentity)
                and (
                    int(entry_status().st_dev),
                    int(entry_status().st_ino),
                    int(entry_status().st_mode),
                    int(entry_status().st_size),
                )
                in {
                    (
                        expected.device,
                        expected.inode,
                        expected.mode,
                        0,
                    ),
                    (
                        expected.device,
                        expected.inode,
                        expected.mode,
                        expected.size,
                    ),
                }
            ):
                raise RuntimeError(f"retired shard {kind} identity changed: {path}")
        if kind == "file":
            os.unlink(path.name, dir_fd=descriptor)
        else:
            os.rmdir(path.name, dir_fd=descriptor)
        os.fsync(descriptor)
        _after_private_trash_entry_removal_boundary(path)
    finally:
        os.close(descriptor)
    if _path_entry_exists(path):
        raise RuntimeError(f"retired shard {kind} was not removed: {path}")


def _unlink_file_if_identity(path: Path, expected: FileIdentity) -> bool:
    return _remove_entry_if_identity(path, expected, kind="file")


def _rmdir_if_identity(path: Path, expected: Sequence[int]) -> bool:
    return _remove_entry_if_identity(path, expected, kind="directory")


def _retire_regular_file_without_pathname_delete(
    path: Path,
    expected: FileIdentity,
    transaction_lock: BinaryIO,
) -> None:
    """Reduce an exact quarantined inode to a durable zero-byte debt record."""

    if _uses_private_trash_retirement():
        _delete_from_private_trash(
            path,
            expected,
            kind="file",
            transaction_lock=transaction_lock,
            allow_truncated_file=True,
        )
        return

    anchor = _read_lock_anchor(transaction_lock, allow_unrecorded_retirement=True)
    for raw in cast(list[object], anchor["retired_entries"]):
        if isinstance(raw, dict) and raw.get("path") == str(path):
            if (
                raw.get("kind") == "file"
                and _identity_payload_matches(path, raw.get("identity"))
                and file_identity(path).size == 0
            ):
                return
            raise RuntimeError(f"retired shard file identity changed: {path}")

    no_follow = getattr(os, "O_NOFOLLOW", 0)
    binary = getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, os.O_RDWR | binary | no_follow)
    try:
        before = os.fstat(descriptor)
        stable = (expected.device, expected.inode, expected.mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or (int(before.st_dev), int(before.st_ino), int(before.st_mode)) != stable
            or int(before.st_nlink) != 1
            or int(before.st_size) not in {0, expected.size}
        ):
            raise RuntimeError(f"retired shard file identity changed: {path}")
        _before_identity_bound_entry_removal(path, "file")
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        retired_identity = FileIdentity(
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_mode),
            int(after.st_size),
            int(after.st_mtime_ns),
        )
    finally:
        os.close(descriptor)
    if file_identity(path) != retired_identity:
        raise RuntimeError(f"retired shard file identity changed: {path}")
    _fsync_directory(path.parent)
    _record_retired_entry(
        transaction_lock,
        path,
        kind="file",
        identity=_identity_payload(retired_identity),
    )


def _retire_empty_directory_without_pathname_delete(
    path: Path,
    expected: Sequence[int],
    transaction_lock: BinaryIO,
) -> None:
    """Persist an exact empty-directory debt record without a pathname rmdir."""

    if _uses_private_trash_retirement():
        _delete_from_private_trash(
            path,
            expected,
            kind="directory",
            transaction_lock=transaction_lock,
        )
        return

    if (
        _path_is_link_like(path)
        or not path.is_dir()
        or _entry_instance(path) != list(expected)
        or any(path.iterdir())
    ):
        raise RuntimeError(f"retired shard directory identity changed: {path}")
    _before_identity_bound_entry_removal(path, "directory")
    if _entry_instance(path) != list(expected) or any(path.iterdir()):
        raise RuntimeError(f"retired shard directory identity changed: {path}")
    _record_retired_entry(
        transaction_lock,
        path,
        kind="directory",
        identity=list(expected),
    )


def _shard_transaction_path(output: Path) -> Path:
    return output / ".shard-build-transaction.json"


def _shard_transaction_lock_path(output: Path) -> Path:
    return output / ".shard-build-transaction.lock"


def _path_is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _is_single_link_regular_file(path: Path) -> bool:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(status.st_mode) and int(status.st_nlink) == 1


def _status_instance(status: os.stat_result) -> list[int]:
    return [int(status.st_dev), int(status.st_ino), int(status.st_mode)]


def _identity_payload(identity: FileIdentity) -> list[int]:
    return [
        identity.device,
        identity.inode,
        identity.mode,
        identity.size,
        identity.mtime_ns,
    ]


def _identity_from_payload(value: object) -> FileIdentity:
    if (
        not isinstance(value, list)
        or len(value) != 5
        or any(type(item) is not int for item in value)
    ):
        raise RuntimeError("durable file identity contract is invalid")
    return FileIdentity(*value)


def _lock_wal_slot_offset(slot: int) -> int:
    if slot < 0 or slot >= _SHARD_LOCK_WAL_SLOT_COUNT:
        raise RuntimeError("shard lock WAL slot is invalid")
    return 1 + slot * _SHARD_LOCK_WAL_SLOT_CAPACITY


def _lock_wal_record(payload: Mapping[str, Any], generation: int) -> bytes:
    encoded = canonical_json_bytes(dict(payload))
    maximum = _SHARD_LOCK_WAL_SLOT_CAPACITY - _SHARD_LOCK_WAL_HEADER.size
    if len(encoded) > maximum:
        raise RuntimeError(
            "shard transaction lock anchor exceeds its fixed WAL capacity"
        )
    generation_bytes = struct.pack(">QQ", generation, len(encoded))
    checksum = hashlib.sha256(generation_bytes + encoded).digest()
    return (
        _SHARD_LOCK_WAL_HEADER.pack(
            _SHARD_LOCK_WAL_MAGIC,
            generation,
            len(encoded),
            checksum,
        )
        + encoded
    )


def _read_lock_wal_slot(
    handle: BinaryIO, slot: int
) -> tuple[int, dict[str, Any]] | None:
    handle.seek(_lock_wal_slot_offset(slot))
    header = handle.read(_SHARD_LOCK_WAL_HEADER.size)
    if len(header) != _SHARD_LOCK_WAL_HEADER.size:
        return None
    try:
        magic, generation, length, checksum = _SHARD_LOCK_WAL_HEADER.unpack(header)
    except struct.error:
        return None
    maximum = _SHARD_LOCK_WAL_SLOT_CAPACITY - _SHARD_LOCK_WAL_HEADER.size
    if (
        magic != _SHARD_LOCK_WAL_MAGIC
        or generation < 1
        or length < 2
        or length > maximum
    ):
        return None
    encoded = handle.read(length)
    if len(encoded) != length:
        return None
    generation_bytes = struct.pack(">QQ", generation, length)
    if hashlib.sha256(generation_bytes + encoded).digest() != checksum:
        return None
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return generation, payload


def _read_lock_storage(
    handle: BinaryIO,
) -> tuple[dict[str, Any], int, int | None]:
    """Read the newest checksummed generation, with legacy v1 migration only."""

    valid: list[tuple[int, int, dict[str, Any]]] = []
    for slot in range(_SHARD_LOCK_WAL_SLOT_COUNT):
        located = _read_lock_wal_slot(handle, slot)
        if located is not None:
            generation, payload = located
            valid.append((generation, slot, payload))
    if valid:
        valid.sort(key=lambda item: item[0], reverse=True)
        if len(valid) > 1 and valid[0][0] == valid[1][0] and valid[0][2] != valid[1][2]:
            raise RuntimeError("shard transaction lock WAL generations conflict")
        if len(valid) > 1 and valid[0][0] - valid[1][0] != 1:
            raise RuntimeError("shard transaction lock WAL generation gap is invalid")
        generation, slot, payload = valid[0]
        return payload, generation, slot

    # A legacy lock stored one unchecksummed JSON object immediately after the
    # byte used for OS locking.  It is consulted only when *no* valid v2 slot
    # exists, so a valid v2 generation can never roll back to legacy bytes.
    handle.seek(1)
    legacy = handle.read(_SHARD_LOCK_WAL_SLOT_CAPACITY - 1)
    if not legacy.lstrip().startswith(b"{"):
        raise RuntimeError("shard transaction lock WAL has no valid generation")
    try:
        payload = json.loads(legacy.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("shard transaction lock owner contract is invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("shard transaction lock owner contract is invalid")
    return payload, 0, None


def _lock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(  # type: ignore[attr-defined]
            handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
        )


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(  # type: ignore[attr-defined]
            handle.fileno(), fcntl.LOCK_UN  # type: ignore[attr-defined]
        )


def _uses_private_trash_retirement() -> bool:
    return os.name != "nt"


def _private_trash_path(output: Path) -> Path:
    return output / _SHARD_PRIVATE_TRASH


def _ensure_private_trash(output: Path) -> dict[str, Any] | None:
    """Create the POSIX retirement boundary without trusting a group-owned root.

    Trust boundary: the operating-system user that owns the lock and this 0700
    directory is trusted.  Group/other writers of ``output`` are not.  All
    pathname deletion is performed relative to an identity-checked descriptor
    for this directory, so no deletion is authorized merely by output-root
    pathname ownership.
    """

    if not _uses_private_trash_retirement():
        return None
    trash = _private_trash_path(output)
    try:
        trash.mkdir(mode=0o700)
        _fsync_directory(output)
    except FileExistsError:
        pass
    if _path_is_link_like(trash):
        raise RuntimeError(f"unsafe shard private trash entry: {trash}")
    try:
        status = trash.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"unsafe shard private trash entry: {trash}") from exc
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"unsafe shard private trash entry: {trash}")
    if os.name != "nt":
        expected_uid = getattr(os, "geteuid", lambda: status.st_uid)()
        if (
            int(status.st_uid) != int(expected_uid)
            or stat.S_IMODE(status.st_mode) != 0o700
        ):
            raise RuntimeError(f"unsafe shard private trash permissions: {trash}")
    return {
        "path": str(trash),
        "identity": _status_instance(status),
    }


def _validate_private_trash_anchor(output: Path, raw: object) -> Path | None:
    if not _uses_private_trash_retirement():
        if raw is not None:
            raise RuntimeError("shard private trash anchor is invalid")
        return None
    trash = _private_trash_path(output)
    if (
        not isinstance(raw, dict)
        or raw.get("path") != str(trash)
        or _path_is_link_like(trash)
    ):
        raise RuntimeError("shard private trash identity changed")
    try:
        status = trash.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("shard private trash identity changed") from exc
    if not stat.S_ISDIR(status.st_mode) or raw.get("identity") != _status_instance(
        status
    ):
        raise RuntimeError("shard private trash identity changed")
    if os.name != "nt":
        expected_uid = getattr(os, "geteuid", lambda: status.st_uid)()
        if (
            int(status.st_uid) != int(expected_uid)
            or stat.S_IMODE(status.st_mode) != 0o700
        ):
            raise RuntimeError("shard private trash identity changed")
    return trash


def _add_authorized_retired_path(
    output: Path, allowed: set[str], raw_path: object, *, kind: str | None = None
) -> None:
    if not isinstance(raw_path, str):
        return
    path = Path(raw_path)
    match = re.fullmatch(r"[0-9a-f]{32}\.(file|directory)", path.name)
    expected_parent = _private_trash_path(output)
    if not _uses_private_trash_retirement():
        match = re.fullmatch(
            r"\.bitguard-retired-[0-9a-f]{32}\.(file|directory)", path.name
        )
        expected_parent = output
    if (
        path.parent != expected_parent
        or match is None
        or (kind is not None and match.group(1) != kind)
    ):
        return
    allowed.add(str(path))


def _add_transaction_retirement_paths(
    output: Path, allowed: set[str], payload: object
) -> None:
    if not isinstance(payload, dict) or payload.get("output_root") != str(output):
        return
    for field in ("cleanup_quarantine_intent", "work_rollback_intent"):
        intent = payload.get(field)
        if isinstance(intent, dict):
            kind = intent.get("kind")
            _add_authorized_retired_path(
                output,
                allowed,
                intent.get("quarantine_path"),
                kind=kind if kind in {"file", "directory"} else None,
            )
    directory_intent = payload.get("directory_rollback_intent")
    if isinstance(directory_intent, dict):
        _add_authorized_retired_path(
            output,
            allowed,
            directory_intent.get("quarantine_path"),
            kind="directory",
        )
    rollback_intent = payload.get("rollback_intent")
    if isinstance(rollback_intent, str):
        _add_authorized_retired_path(
            output,
            allowed,
            str(
                _public_file_retirement_path(
                    output, payload.get("transaction_id"), rollback_intent
                )
            ),
            kind="file",
        )


def _authorized_unrecorded_retirement_paths(
    output: Path, anchor: Mapping[str, Any]
) -> set[str]:
    """Return exact quarantine paths named by active durable intents only."""

    allowed: set[str] = set()
    owner = anchor.get("owner")
    if isinstance(owner, dict) and owner.get("phase") == "retiring":
        record = owner.get("record")
        if isinstance(record, dict):
            _add_authorized_retired_path(
                output, allowed, record.get("quarantine_path"), kind="file"
            )
    initializers = anchor.get("retiring_initializers")
    if isinstance(initializers, list):
        for raw in initializers:
            if isinstance(raw, dict):
                _add_authorized_retired_path(
                    output, allowed, raw.get("debt_path"), kind="file"
                )
    temporaries = anchor.get("temporaries")
    if isinstance(temporaries, list):
        for raw in temporaries:
            if isinstance(raw, dict) and raw.get("phase") == "retiring":
                _add_authorized_retired_path(
                    output, allowed, raw.get("quarantine_path"), kind="file"
                )
    manifest = anchor.get("manifest")
    if isinstance(manifest, dict) and manifest.get("phase") == "retiring":
        quarantine_paths = manifest.get("quarantine_paths")
        if isinstance(quarantine_paths, dict):
            for raw_path in quarantine_paths.values():
                _add_authorized_retired_path(output, allowed, raw_path, kind="file")
    journal = anchor.get("journal")
    if isinstance(journal, dict):
        if journal.get("phase") == "retiring":
            retirement = journal.get("retirement")
            if isinstance(retirement, dict):
                _add_authorized_retired_path(
                    output,
                    allowed,
                    retirement.get("quarantine_path"),
                    kind="file",
                )

    transaction_path = _shard_transaction_path(output)
    try:
        transaction_payload = json.loads(transaction_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        transaction_payload = None
    accepted = journal.get("accepted", []) if isinstance(journal, dict) else []
    if isinstance(transaction_payload, dict):
        accepted_current = any(
            _record_matches_payload(transaction_path, record, transaction_payload)
            for record in accepted
        )
        if accepted_current:
            _add_transaction_retirement_paths(output, allowed, transaction_payload)

    journal_phase = journal.get("phase") if isinstance(journal, dict) else None
    replacement = journal.get("replacement") if isinstance(journal, dict) else None
    if isinstance(replacement, dict):
        old = replacement.get("old")
        old_identity = old.get("identity") if isinstance(old, dict) else None
        new_payload = replacement.get("new_payload")
        stable_instance = replacement.get("stable_instance")
        replacement_is_anchored = (
            journal_phase in {"replace_intended", "rewrite_started"}
            and replacement.get("journal_path") == str(transaction_path)
            and isinstance(old, dict)
            and accepted == [old]
            and isinstance(old.get("fingerprint"), str)
            and isinstance(old_identity, list)
            and len(old_identity) == 5
            and not any(type(value) is not int for value in old_identity)
            and isinstance(new_payload, dict)
            and replacement.get("new_fingerprint") == stable_fingerprint(new_payload)
            and isinstance(stable_instance, list)
            and len(stable_instance) == 3
            and not any(type(value) is not int for value in stable_instance)
            and old_identity[:3] == stable_instance
        )
        if replacement_is_anchored:
            # ``new_payload`` is authenticated by the checksummed lock WAL and
            # the accepted old-record transition.  It remains authoritative
            # while the in-place journal pathname is temporarily torn.
            _add_transaction_retirement_paths(output, allowed, new_payload)
    return allowed


def _read_lock_anchor(
    handle: BinaryIO, *, allow_unrecorded_retirement: bool = False
) -> dict[str, Any]:
    try:
        payload, generation, _ = _read_lock_storage(handle)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError("shard transaction lock owner contract is invalid") from exc
    status = os.fstat(handle.fileno())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _SHARD_LOCK_SCHEMA
        or payload.get("lock_instance") != _status_instance(status)
        or not isinstance(payload.get("owner_token"), str)
        or len(str(payload["owner_token"])) != 32
        or not isinstance(payload.get("lock_path"), str)
        or not isinstance(payload.get("temporaries"), list)
        or not isinstance(payload.get("retiring_initializers", []), list)
        or not isinstance(payload.get("retired_entries", []), list)
    ):
        raise RuntimeError("shard transaction lock identity contract is invalid")
    payload.setdefault("retiring_initializers", [])
    payload.setdefault("retired_entries", [])
    # The compatibility flag no longer grants blanket authority.  Recovery may
    # observe unrecorded debt only at an exact path named by an active WAL intent.
    del allow_unrecorded_retirement
    output = Path(str(payload["lock_path"])).parent
    if generation == 0 and payload.get("private_trash") is None:
        _ensure_private_trash(output)
    else:
        _validate_private_trash_anchor(output, payload.get("private_trash"))
    _validate_retired_entries(
        output,
        payload["retired_entries"],
        allowed_unrecorded=_authorized_unrecorded_retirement_paths(output, payload),
    )
    return payload


def _assert_lock_anchor_path(handle: BinaryIO, anchor: Mapping[str, Any]) -> None:
    path = Path(str(anchor["lock_path"]))
    status = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(status.st_mode)
        or int(status.st_nlink) != 1
        or _path_is_link_like(path)
        or _entry_instance(path) != _status_instance(status)
    ):
        raise RuntimeError(f"shard transaction lock identity changed: {path}")


def _write_lock_anchor(handle: BinaryIO, anchor: Mapping[str, Any]) -> None:
    _assert_lock_anchor_path(handle, anchor)
    output = Path(str(anchor["lock_path"])).parent
    anchor_payload = dict(anchor)
    anchor_payload["private_trash"] = _ensure_private_trash(output)
    _validate_retired_entries(
        output,
        anchor_payload.get("retired_entries", []),
        allowed_unrecorded=_authorized_unrecorded_retirement_paths(
            output, anchor_payload
        ),
    )
    # Capacity is checked before seeking or writing so an oversized anchor
    # cannot damage either accepted generation.
    _lock_wal_record(anchor_payload, 1)
    _, generation, current_slot = _read_lock_storage(handle)
    next_generation = generation + 1
    encoded = _lock_wal_record(anchor_payload, next_generation)
    next_slot = 1 if current_slot in {None, 0} else 0
    handle.seek(_lock_wal_slot_offset(next_slot))
    _write_all(handle, encoded)
    handle.flush()
    os.fsync(handle.fileno())
    observed, observed_generation, observed_slot = _read_lock_storage(handle)
    if (
        observed_generation != next_generation
        or observed_slot != next_slot
        or observed != anchor_payload
    ):
        raise RuntimeError("shard transaction lock WAL write was not durable")


def _retired_entry_path(output: Path, kind: str) -> Path:
    if _uses_private_trash_retirement():
        return _private_trash_path(output) / f"{uuid.uuid4().hex}.{kind}"
    return output / f".bitguard-retired-{uuid.uuid4().hex}.{kind}"


def _retirement_path_matches(output: Path, path: Path, *, kind: str) -> bool:
    if kind not in {"file", "directory"}:
        return False
    if _uses_private_trash_retirement():
        return (
            path.parent == _private_trash_path(output)
            and re.fullmatch(rf"[0-9a-f]{{32}}\.{re.escape(kind)}", path.name)
            is not None
        )
    return (
        path.parent == output
        and re.fullmatch(
            rf"\.bitguard-retired-[0-9a-f]{{32}}\.{re.escape(kind)}", path.name
        )
        is not None
    )


def _validate_retired_entries(
    output: Path, raw_records: object, *, allowed_unrecorded: set[str]
) -> None:
    if not isinstance(raw_records, list):
        raise RuntimeError("shard retired-entry contract is invalid")
    observed_paths: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise RuntimeError("shard retired-entry contract is invalid")
        path = Path(str(raw.get("path", "")))
        kind = raw.get("kind")
        identity = raw.get("identity")
        if (
            path.parent != output
            or re.fullmatch(
                r"\.bitguard-retired-[0-9a-f]{32}\.(?:file|directory)",
                path.name,
            )
            is None
            or kind not in {"file", "directory"}
            or path.suffix != f".{kind}"
            or str(path) in observed_paths
        ):
            raise RuntimeError("shard retired-entry contract is invalid")
        observed_paths.add(str(path))
        if not _path_entry_exists(path):
            continue
        if kind == "file":
            if (
                not isinstance(identity, list)
                or len(identity) != 5
                or any(type(value) is not int for value in identity)
                or _path_is_link_like(path)
                or not _is_single_link_regular_file(path)
                or not _identity_payload_matches(path, identity)
                or file_identity(path).size != 0
            ):
                raise RuntimeError(f"retired shard file identity changed: {path}")
        elif (
            not isinstance(identity, list)
            or len(identity) != 3
            or any(type(value) is not int for value in identity)
            or _path_is_link_like(path)
            or not path.is_dir()
            or _entry_instance(path) != identity
            or any(path.iterdir())
        ):
            raise RuntimeError(f"retired shard directory identity changed: {path}")
    if _uses_private_trash_retirement():
        record = _ensure_private_trash(output)
        trash = _validate_private_trash_anchor(output, record)
        assert trash is not None
        count = 0
        for path in trash.iterdir():
            count += 1
            if count > _SHARD_PRIVATE_TRASH_MAX_ENTRIES:
                raise RuntimeError("shard private trash entry bound exceeded")
            if str(path) not in observed_paths and str(path) not in allowed_unrecorded:
                raise RuntimeError(f"unrecorded retired shard entry: {path}")
    else:
        for path in output.glob(".bitguard-retired-*"):
            if str(path) not in observed_paths and str(path) not in allowed_unrecorded:
                raise RuntimeError(f"unrecorded retired shard entry: {path}")


def _record_retired_entry(
    transaction_lock: BinaryIO,
    path: Path,
    *,
    kind: str,
    identity: Sequence[int],
) -> None:
    anchor = _read_lock_anchor(transaction_lock, allow_unrecorded_retirement=True)
    records = cast(list[object], anchor["retired_entries"])
    record = {"path": str(path), "kind": kind, "identity": list(identity)}
    if record not in records:
        records.append(record)
    _validate_retired_entries(
        Path(str(anchor["lock_path"])).parent,
        records,
        allowed_unrecorded=_authorized_unrecorded_retirement_paths(
            Path(str(anchor["lock_path"])).parent, anchor
        ),
    )
    anchor["retired_entries"] = records
    _write_lock_anchor(transaction_lock, anchor)


def _recorded_retired_entry(
    transaction_lock: BinaryIO,
    path: Path,
    *,
    kind: str,
) -> bool:
    anchor = _read_lock_anchor(transaction_lock, allow_unrecorded_retirement=True)
    for raw in cast(list[object], anchor["retired_entries"]):
        if not isinstance(raw, dict) or raw.get("path") != str(path):
            continue
        if raw.get("kind") != kind:
            raise RuntimeError(f"retired shard {kind} identity changed: {path}")
        identity = raw.get("identity")
        if not _path_entry_exists(path):
            return os.name == "nt"
        if kind == "file":
            return _identity_payload_matches(path, identity)
        return _entry_instance(path) == identity
    return False


def _retire_quarantined_entry(
    path: Path,
    expected: FileIdentity | Sequence[int],
    *,
    kind: str,
    transaction_lock: BinaryIO,
) -> str:
    """Finish an identity-bound quarantine as gone (Windows) or durable debt."""

    if not _path_entry_exists(path):
        if _uses_private_trash_retirement():
            anchor = _read_lock_anchor(
                transaction_lock, allow_unrecorded_retirement=True
            )
            output = Path(str(anchor["lock_path"])).parent
            if _retirement_path_matches(output, path, kind=kind) and str(
                path
            ) in _authorized_unrecorded_retirement_paths(output, anchor):
                return "gone"
        elif os.name == "nt":
            return "gone"
        raise RuntimeError(f"quarantined shard {kind} disappeared: {path}")
    if _recorded_retired_entry(transaction_lock, path, kind=kind):
        return "debt_recorded"
    if not _entry_matches_expected(path, expected, kind=kind):
        raise RuntimeError(f"quarantined shard {kind} identity changed: {path}")
    if os.name == "nt" and not _uses_private_trash_retirement():
        _remove_entry_if_identity(path, expected, kind=kind)
        _fsync_directory(path.parent)
        return "gone"
    if kind == "file":
        _retire_regular_file_without_pathname_delete(
            path, cast(FileIdentity, expected), transaction_lock
        )
    else:
        _retire_empty_directory_without_pathname_delete(
            path, cast(Sequence[int], expected), transaction_lock
        )
    return "gone" if _uses_private_trash_retirement() else "debt_recorded"


def _lock_initializing_paths(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f".{path.name}.*.initializing"))


def _lock_retiring_paths(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f".{path.name}.*.retiring"))


def _read_initialized_lock(
    path: Path, lock_path: Path
) -> tuple[dict[str, Any], FileIdentity]:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    binary = getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | binary | no_follow)
    except OSError as exc:
        raise RuntimeError(f"unsafe shard lock initialization entry: {path}") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or int(status.st_nlink) not in {1, 2}:
            raise RuntimeError(f"unsafe shard lock initialization entry: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            handle.seek(0)
            if handle.read(1) != b"\0":
                raise RuntimeError(f"shard lock initialization is incomplete: {path}")
            try:
                payload, _, _ = _read_lock_storage(handle)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"shard lock initialization is incomplete: {path}"
                ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _SHARD_LOCK_SCHEMA
            or payload.get("lock_instance") != _status_instance(status)
            or payload.get("lock_path") != str(lock_path)
            or not isinstance(payload.get("owner_token"), str)
            or len(str(payload["owner_token"])) != 32
            or payload.get("journal") is not None
            or payload.get("owner") is not None
            or payload.get("manifest") is not None
            or payload.get("temporaries") != []
            or payload.get("retiring_initializers", []) != []
            or payload.get("retired_entries", []) != []
        ):
            raise RuntimeError(f"shard lock initialization identity changed: {path}")
        return payload, FileIdentity(
            int(status.st_dev),
            int(status.st_ino),
            int(status.st_mode),
            int(status.st_size),
            int(status.st_mtime_ns),
        )
    finally:
        os.close(descriptor)


def _is_torn_lock_initializer(path: Path) -> bool:
    """Recognize structurally v2 torn bytes without ever accepting their payload."""

    try:
        if _path_is_link_like(path) or not _is_single_link_regular_file(path):
            return False
        with path.open("rb") as handle:
            if handle.read(1) != b"\0":
                return False
            try:
                _read_lock_storage(handle)
            except RuntimeError:
                pass
            else:
                return False
            size = path.stat().st_size
            if size > 1 + _SHARD_LOCK_WAL_SLOT_COUNT * _SHARD_LOCK_WAL_SLOT_CAPACITY:
                return False
            for slot in range(_SHARD_LOCK_WAL_SLOT_COUNT):
                handle.seek(_lock_wal_slot_offset(slot))
                header = handle.read(_SHARD_LOCK_WAL_HEADER.size)
                if len(header) < _SHARD_LOCK_WAL_HEADER.size:
                    if slot == 0 and _SHARD_LOCK_WAL_MAGIC.startswith(header):
                        return True
                    continue
                magic, generation, length, _ = _SHARD_LOCK_WAL_HEADER.unpack(header)
                if (
                    magic == _SHARD_LOCK_WAL_MAGIC
                    and generation >= 1
                    and 2
                    <= length
                    <= _SHARD_LOCK_WAL_SLOT_CAPACITY - _SHARD_LOCK_WAL_HEADER.size
                ):
                    return True
            return False
    except (OSError, struct.error):
        return False


def _after_lock_initializing_boundary(_initializing: Path) -> None:
    """Test seam after a lock inode is fully initialized but remains private."""


def _after_lock_publish_link_boundary(_initializing: Path, _lock: Path) -> None:
    """Test seam after no-replace lock publication and before private-name removal."""


def _after_lock_loser_retirement_boundary(_initializing: Path, _retiring: Path) -> None:
    """Test seam after a losing initializer has a durable retirement name."""


def _before_lock_initializer_creation(_lock: Path) -> None:
    """Test seam after observing no initializer and before creating a unique one."""


def _lock_retiring_path(initializing: Path) -> Path:
    suffix = ".initializing"
    if not initializing.name.endswith(suffix):
        raise RuntimeError(f"invalid shard lock initialization path: {initializing}")
    return initializing.with_name(f"{initializing.name[:-len(suffix)]}.retiring")


def _finish_losing_lock_retirement(path: Path, transaction_lock: BinaryIO) -> None:
    anchor = _read_lock_anchor(transaction_lock)
    records = anchor.get("retiring_initializers")
    if not isinstance(records, list) or len(records) > 1:
        raise RuntimeError("shard lock retirement anchor is invalid")
    if not records:
        if _lock_retiring_paths(path):
            raise RuntimeError("unanchored shard lock retirement entry")
        return
    record = records[0]
    if not isinstance(record, dict):
        raise RuntimeError("shard lock retirement anchor is invalid")
    initializing = Path(str(record.get("initializing_path", "")))
    retiring = Path(str(record.get("retiring_path", "")))
    debt = Path(str(record.get("debt_path", "")))
    if not record.get("debt_path"):
        debt = _retired_entry_path(path.parent, "file")
        record = {**record, "debt_path": str(debt)}
        anchor["retiring_initializers"] = [record]
        _write_lock_anchor(transaction_lock, anchor)
    expected = _identity_from_payload(record.get("identity"))
    if (
        initializing.parent != path.parent
        or retiring != _lock_retiring_path(initializing)
        or not _retirement_path_matches(path.parent, debt, kind="file")
        or record.get("phase")
        not in {
            "intended",
            "quarantined",
            "debt_quarantined",
            "gone",
            "debt_recorded",
        }
    ):
        raise RuntimeError("shard lock retirement anchor is invalid")
    lock_identity = file_identity(path)
    if expected == lock_identity:
        raise RuntimeError(f"shard lock retirement identity changed: {retiring}")
    if record["phase"] == "intended":
        if _path_entry_exists(initializing):
            _, observed = _read_initialized_lock(initializing, path)
            if observed != expected or int(os.lstat(initializing).st_nlink) != 1:
                raise RuntimeError(
                    f"shard lock retirement identity changed: {initializing}"
                )
            if _path_entry_exists(retiring):
                raise RuntimeError(
                    f"shard lock retirement path already exists: {retiring}"
                )
            _rename_directory_no_replace(initializing, retiring)
            _fsync_directory(path.parent)
        elif not _path_entry_exists(retiring):
            raise RuntimeError(
                f"shard lock retirement identity changed: {initializing}"
            )
        try:
            _, observed = _read_initialized_lock(retiring, path)
        except RuntimeError:
            if not _path_entry_exists(initializing):
                try:
                    _rename_directory_no_replace(retiring, initializing)
                    _fsync_directory(path.parent)
                except BaseException:
                    pass
            raise
        if observed != expected or int(os.lstat(retiring).st_nlink) != 1:
            if not _path_entry_exists(initializing):
                try:
                    _rename_directory_no_replace(retiring, initializing)
                    _fsync_directory(path.parent)
                except BaseException:
                    pass
            raise RuntimeError(f"shard lock retirement identity changed: {retiring}")
        record = {**record, "phase": "quarantined"}
        anchor = _read_lock_anchor(transaction_lock)
        anchor["retiring_initializers"] = [record]
        _write_lock_anchor(transaction_lock, anchor)
    if _path_entry_exists(initializing):
        raise RuntimeError(f"shard lock retirement path was replaced: {initializing}")
    if record["phase"] == "quarantined" and _path_entry_exists(retiring):
        _, observed = _read_initialized_lock(retiring, path)
        if observed != expected or int(os.lstat(retiring).st_nlink) != 1:
            raise RuntimeError(f"shard lock retirement identity changed: {retiring}")
        _after_lock_loser_retirement_boundary(initializing, retiring)
        if file_identity(retiring) != expected or file_identity(path) != lock_identity:
            raise RuntimeError(f"shard lock retirement identity changed: {retiring}")
        _move_entry_to_quarantine_exact(retiring, debt, expected, kind="file")
        record = {**record, "phase": "debt_quarantined"}
        anchor = _read_lock_anchor(transaction_lock, allow_unrecorded_retirement=True)
        anchor["retiring_initializers"] = [record]
        _write_lock_anchor(transaction_lock, anchor)
    elif record["phase"] == "quarantined":
        if not _entry_matches_expected(debt, expected, kind="file"):
            if not _recorded_retired_entry(transaction_lock, debt, kind="file"):
                raise RuntimeError(
                    f"shard lock retirement identity changed: {retiring}"
                )
        record = {**record, "phase": "debt_quarantined"}
        anchor = _read_lock_anchor(transaction_lock, allow_unrecorded_retirement=True)
        anchor["retiring_initializers"] = [record]
        _write_lock_anchor(transaction_lock, anchor)
    if record["phase"] == "debt_quarantined":
        outcome = _retire_quarantined_entry(
            debt,
            expected,
            kind="file",
            transaction_lock=transaction_lock,
        )
        record = {**record, "phase": outcome}
        anchor = _read_lock_anchor(transaction_lock)
        anchor["retiring_initializers"] = [record]
        _write_lock_anchor(transaction_lock, anchor)
    if record["phase"] == "gone":
        if not (
            os.name == "nt" or _uses_private_trash_retirement()
        ) or _path_entry_exists(debt):
            raise RuntimeError(f"shard lock retirement identity changed: {debt}")
    elif record["phase"] == "debt_recorded":
        if not _recorded_retired_entry(transaction_lock, debt, kind="file"):
            raise RuntimeError(f"shard lock retirement identity changed: {debt}")
    else:
        raise RuntimeError("shard lock retirement anchor is invalid")
    anchor = _read_lock_anchor(transaction_lock)
    if anchor.get("retiring_initializers") != [record]:
        raise RuntimeError("shard lock retirement anchor changed")
    anchor["retiring_initializers"] = []
    _write_lock_anchor(transaction_lock, anchor)


def _retire_losing_lock_initializer(
    path: Path,
    initializing: Path,
    expected: FileIdentity,
    transaction_lock: BinaryIO,
) -> None:
    retiring = _lock_retiring_path(initializing)
    anchor = _read_lock_anchor(transaction_lock)
    if anchor.get("retiring_initializers"):
        raise RuntimeError("another shard lock initializer retirement is active")
    anchor["retiring_initializers"] = [
        {
            "phase": "intended",
            "initializing_path": str(initializing),
            "retiring_path": str(retiring),
            "debt_path": str(_retired_entry_path(path.parent, "file")),
            "identity": _identity_payload(expected),
        }
    ]
    _write_lock_anchor(transaction_lock, anchor)
    _finish_losing_lock_retirement(path, transaction_lock)


def _publish_initialized_lock(path: Path) -> None:
    if _path_entry_exists(path):
        return
    candidates = _lock_initializing_paths(path)
    initializing: Path | None = None
    expected: FileIdentity | None = None
    for candidate in candidates:
        try:
            _, candidate_identity = _read_initialized_lock(candidate, path)
        except RuntimeError:
            if _is_torn_lock_initializer(candidate):
                continue
            raise
        initializing = candidate
        expected = candidate_identity
        break
    if initializing is None:
        _before_lock_initializer_creation(path)
        initializing = path.with_name(f".{path.name}.{uuid.uuid4().hex}.initializing")
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        binary = getattr(os, "O_BINARY", 0)
        descriptor = os.open(
            initializing,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | binary | no_follow,
            0o600,
        )
        try:
            status = os.fstat(descriptor)
            anchor: dict[str, Any] = {
                "schema_version": _SHARD_LOCK_SCHEMA,
                "lock_instance": _status_instance(status),
                "lock_path": str(path),
                "owner_token": uuid.uuid4().hex,
                "journal": None,
                "owner": None,
                "manifest": None,
                "temporaries": [],
                "retiring_initializers": [],
                "retired_entries": [],
                "private_trash": _ensure_private_trash(path.parent),
            }
            encoded = _lock_wal_record(anchor, 1)
            _write_all_descriptor(descriptor, b"\0")
            os.lseek(descriptor, _lock_wal_slot_offset(0), os.SEEK_SET)
            _write_all_descriptor(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
        _, expected = _read_initialized_lock(initializing, path)
        _after_lock_initializing_boundary(initializing)
    else:
        _after_lock_initializing_boundary(initializing)
    assert expected is not None
    try:
        _move_entry_to_quarantine_exact(initializing, path, expected, kind="file")
    except RuntimeError:
        if _path_entry_exists(path):
            return
        raise
    _after_lock_publish_link_boundary(initializing, path)
    if file_identity(path) != expected or _path_entry_exists(initializing):
        raise RuntimeError(
            f"shard lock initialization identity changed: {initializing}"
        )


def _cleanup_lock_initializing_paths(path: Path, transaction_lock: BinaryIO) -> None:
    _finish_losing_lock_retirement(path, transaction_lock)
    lock_identity = file_identity(path)
    for initializing in _lock_initializing_paths(path):
        try:
            observed = file_identity(initializing)
            if observed == lock_identity:
                expected = observed
            else:
                _, expected = _read_initialized_lock(initializing, path)
        except (OSError, RuntimeError) as exc:
            if _is_torn_lock_initializer(initializing):
                # A short initializer has no durable ownership proof.  Preserve
                # it, but do not let it prevent a separately valid lock winner.
                continue
            raise RuntimeError(
                f"shard lock initialization identity changed: {initializing}"
            ) from exc
        if _path_is_link_like(initializing):
            raise RuntimeError(
                f"shard lock initialization identity changed: {initializing}"
            )
        if expected != lock_identity:
            _retire_losing_lock_initializer(
                path, initializing, expected, transaction_lock
            )
            continue
        # Pre-no-replace versions could leave a second hard-link name for the
        # published lock.  There is no portable identity-bound unlink for that
        # alias, so fail closed and preserve it for explicit recovery.
        raise RuntimeError(
            f"legacy shard lock publication alias requires recovery: {initializing}"
        )
    _finish_losing_lock_retirement(path, transaction_lock)


def _acquire_shard_transaction_lock(path: Path) -> BinaryIO:
    _publish_initialized_lock(path)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    binary = getattr(os, "O_BINARY", 0)
    base_flags = os.O_RDWR | binary | no_follow
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or int(before.st_nlink) not in {1, 2}
            or _path_is_link_like(path)
        ):
            raise RuntimeError(f"unsafe shard transaction lock entry: {path}")
        descriptor = os.open(path, base_flags)
        after = os.fstat(descriptor)
        if (
            _status_instance(before) != _status_instance(after)
            or not stat.S_ISREG(after.st_mode)
            or int(after.st_nlink) not in {1, 2}
            or int(after.st_size) < 2
        ):
            os.close(descriptor)
            raise RuntimeError(f"shard transaction lock identity changed: {path}")
    except OSError as exc:
        raise RuntimeError(f"unsafe shard transaction lock entry: {path}") from exc

    handle = os.fdopen(descriptor, "r+b", buffering=0)
    locked = False
    try:
        try:
            _lock_file(handle)
            locked = True
        except (OSError, ImportError) as exc:
            raise RuntimeError(
                f"another shard builder owns the output transaction: {path.parent}"
            ) from exc
        anchor = _read_lock_anchor(handle)
        if anchor.get("lock_path") != str(path):
            raise RuntimeError(f"shard transaction lock path changed: {path}")
        _cleanup_lock_initializing_paths(path, handle)
        _assert_lock_anchor_path(handle, anchor)
        return handle
    except BaseException:
        if locked:
            try:
                _unlock_file(handle)
            except BaseException:
                pass
        handle.close()
        raise


def _release_shard_transaction_lock(handle: BinaryIO) -> None:
    try:
        _unlock_file(handle)
    finally:
        handle.close()


def _anchored_file_record(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": _file_identity_payload(path),
        "fingerprint": stable_fingerprint(dict(payload)),
    }


def _record_matches_payload(
    path: Path, record: object, payload: Mapping[str, Any]
) -> bool:
    return (
        isinstance(record, dict)
        and _identity_payload_matches(path, record.get("identity"))
        and record.get("fingerprint") == stable_fingerprint(dict(payload))
    )


def _write_transaction_temporary(
    path: Path,
    payload: Mapping[str, Any],
    transaction_lock: BinaryIO,
) -> tuple[Path, dict[str, Any]]:
    anchor = _read_lock_anchor(transaction_lock)
    if anchor.get("temporaries"):
        raise RuntimeError("owned shard transaction temporary is still active")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    intent: dict[str, Any] = {
        "phase": "initializing",
        "path": str(temporary),
        "fingerprint": stable_fingerprint(dict(payload)),
    }
    anchor["temporaries"] = [intent]
    _write_lock_anchor(transaction_lock, anchor)
    _after_transaction_temporary_intent_boundary(temporary)
    with temporary.open("xb") as handle:
        created = {
            **intent,
            "phase": "created",
            "instance": _status_instance(os.fstat(handle.fileno())),
        }
        anchor = _read_lock_anchor(transaction_lock)
        anchor["temporaries"] = [created]
        _write_lock_anchor(transaction_lock, anchor)
        _after_transaction_temporary_created_anchor_boundary(temporary)
        _write_all(handle, encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    _after_transaction_temporary_payload_boundary(temporary)
    record = {
        "phase": "active",
        "path": str(temporary),
        **_anchored_file_record(temporary, payload),
    }
    anchor = _read_lock_anchor(transaction_lock)
    anchor["temporaries"] = [record]
    _write_lock_anchor(transaction_lock, anchor)
    return temporary, record


def _after_transaction_temporary_intent_boundary(_temporary: Path) -> None:
    """Test seam after a unique temporary name is durably intended."""


def _after_transaction_temporary_created_anchor_boundary(_temporary: Path) -> None:
    """Test seam after an empty temporary inode is durably identity-anchored."""


def _after_transaction_temporary_payload_boundary(_temporary: Path) -> None:
    """Test seam after payload fsync and before its final identity anchor."""


def _before_shard_transaction_replace(_temporary: Path) -> None:
    """Test seam after a replacement is durably anchored but before publication."""


def _before_journal_inplace_rewrite_boundary(_journal: Path) -> None:
    """Test seam after rewrite intent and before opening the anchored inode."""


def _before_journal_inplace_payload_write_boundary(_journal: Path) -> None:
    """Test seam immediately before the final link-count check and first write."""


def _after_journal_inplace_rewrite_boundary(_journal: Path) -> None:
    """Test seam after exact journal fsync and before activating new metadata."""


def _journal_replacement_record(
    journal: Path, transaction_lock: BinaryIO
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    anchor = _read_lock_anchor(transaction_lock)
    journal_anchor = anchor.get("journal")
    if not isinstance(journal_anchor, dict) or journal_anchor.get("phase") not in {
        "replace_intended",
        "rewrite_started",
    }:
        return None
    replacement = journal_anchor.get("replacement")
    accepted = journal_anchor.get("accepted")
    if not isinstance(replacement, dict) or not isinstance(accepted, list):
        raise RuntimeError("shard journal replacement anchor is invalid")
    old_record = replacement.get("old")
    new_payload = replacement.get("new_payload")
    stable_instance = replacement.get("stable_instance")
    old_identity = old_record.get("identity") if isinstance(old_record, dict) else None
    if (
        replacement.get("journal_path") != str(journal)
        or not isinstance(old_record, dict)
        or accepted != [old_record]
        or not isinstance(new_payload, dict)
        or replacement.get("new_fingerprint") != stable_fingerprint(new_payload)
        or not isinstance(stable_instance, list)
        or len(stable_instance) != 3
        or any(type(value) is not int for value in stable_instance)
        or not isinstance(old_identity, list)
        or len(old_identity) != 5
        or old_identity[:3] != stable_instance
    ):
        raise RuntimeError("shard journal replacement anchor is invalid")
    return anchor, replacement


def _complete_journal_replacement(
    journal: Path, transaction_lock: BinaryIO
) -> FileIdentity:
    located = _journal_replacement_record(journal, transaction_lock)
    if located is None:
        raise RuntimeError("shard journal replacement anchor is missing")
    anchor, replacement = located
    journal_anchor = cast(dict[str, Any], anchor["journal"])
    phase = str(journal_anchor["phase"])
    old_record = cast(dict[str, Any], replacement["old"])
    new_payload = cast(dict[str, Any], replacement["new_payload"])
    stable_instance = cast(list[int], replacement["stable_instance"])

    if phase == "replace_intended":
        try:
            current = _read_json_object(journal, "transaction")
        except RuntimeError:
            raise RuntimeError(
                f"shard build transaction identity changed: {journal}"
            ) from None
        if not _record_matches_payload(journal, old_record, current):
            raise RuntimeError(f"shard build transaction identity changed: {journal}")
        journal_anchor["phase"] = "rewrite_started"
        anchor["journal"] = journal_anchor
        _write_lock_anchor(transaction_lock, anchor)
        phase = "rewrite_started"

    if phase != "rewrite_started":
        raise RuntimeError("shard journal replacement phase is invalid")
    if _entry_instance(journal) != stable_instance or _path_is_link_like(journal):
        raise RuntimeError(f"shard build transaction identity changed: {journal}")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    binary = getattr(os, "O_BINARY", 0)
    _before_journal_inplace_rewrite_boundary(journal)
    descriptor = os.open(journal, os.O_RDWR | binary | no_follow)
    try:
        before = os.fstat(descriptor)
        if (
            _status_instance(before) != stable_instance
            or not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or _entry_instance(journal) != stable_instance
        ):
            raise RuntimeError(f"shard build transaction identity changed: {journal}")
        encoded = canonical_json_bytes(new_payload) + b"\n"
        _before_journal_inplace_payload_write_boundary(journal)
        immediately_before_write = os.fstat(descriptor)
        if (
            _status_instance(immediately_before_write) != stable_instance
            or int(immediately_before_write.st_nlink) != 1
            or _entry_instance(journal) != stable_instance
        ):
            raise RuntimeError(f"shard build transaction identity changed: {journal}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        _write_all_descriptor(descriptor, encoded)
        os.ftruncate(descriptor, len(encoded))
        os.fsync(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        _status_instance(after) != stable_instance
        or int(after.st_nlink) != 1
        or _entry_instance(journal) != stable_instance
    ):
        raise RuntimeError(f"shard build transaction identity changed: {journal}")
    _fsync_directory(journal.parent)
    _after_journal_inplace_rewrite_boundary(journal)
    observed = _read_json_object(journal, "transaction")
    if observed != new_payload:
        raise RuntimeError(f"shard build transaction identity changed: {journal}")
    new_record = _anchored_file_record(journal, observed)
    anchor = _read_lock_anchor(transaction_lock)
    anchor["journal"] = {"phase": "active", "accepted": [new_record]}
    anchor["temporaries"] = []
    _write_lock_anchor(transaction_lock, anchor)
    return file_identity(journal)


def _write_shard_transaction(
    path: Path, payload: Mapping[str, Any], transaction_lock: BinaryIO
) -> FileIdentity:
    if _path_entry_exists(path):
        raise RuntimeError(f"shard build transaction already exists: {path}")
    temporary, record = _write_transaction_temporary(path, payload, transaction_lock)
    anchor = _read_lock_anchor(transaction_lock)
    anchor["journal"] = {"phase": "transition", "accepted": [record]}
    _write_lock_anchor(transaction_lock, anchor)
    _before_shard_transaction_replace(temporary)
    expected = _identity_from_payload(record["identity"])
    _move_entry_to_quarantine_exact(temporary, path, expected, kind="file")
    identity = file_identity(path)
    if _identity_payload(identity) != record["identity"]:
        raise RuntimeError(f"shard build transaction identity changed: {path}")
    anchor = _read_lock_anchor(transaction_lock)
    anchor["journal"] = {"phase": "active", "accepted": [record]}
    anchor["temporaries"] = []
    _write_lock_anchor(transaction_lock, anchor)
    return identity


def _replace_shard_transaction(
    path: Path,
    payload: Mapping[str, Any],
    expected_identity: FileIdentity,
    transaction_lock: BinaryIO,
) -> FileIdentity:
    if not _path_entry_exists(path) or file_identity(path) != expected_identity:
        raise RuntimeError(f"shard build transaction identity changed: {path}")
    anchor = _read_lock_anchor(transaction_lock)
    current = anchor.get("journal")
    if not isinstance(current, dict) or current.get("phase") != "active":
        raise RuntimeError("shard build transaction anchor is not active")
    old_record = _anchored_file_record(path, _read_json_object(path, "transaction"))
    if not any(
        _record_matches_payload(path, record, _read_json_object(path, "transaction"))
        for record in current.get("accepted", [])
    ):
        raise RuntimeError(f"shard build transaction identity changed: {path}")
    anchor = _read_lock_anchor(transaction_lock)
    anchor["journal"] = {
        "phase": "replace_intended",
        "accepted": [old_record],
        "replacement": {
            "journal_path": str(path),
            "old": old_record,
            "stable_instance": _identity_payload(expected_identity)[:3],
            "new_payload": dict(payload),
            "new_fingerprint": stable_fingerprint(dict(payload)),
        },
    }
    _write_lock_anchor(transaction_lock, anchor)
    _before_shard_transaction_replace(path)
    return _complete_journal_replacement(path, transaction_lock)


def _read_json_object(path: Path, subject: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"shard {subject} cannot be validated: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"shard {subject} is not a JSON object")
    return payload


def _write_shard_owner_marker(
    owner_marker: Path,
    initializing_marker: Path,
    payload: Mapping[str, Any],
    transaction_lock: BinaryIO,
) -> FileIdentity:
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    record: dict[str, Any] = {
        "initializing_path": str(initializing_marker),
        "owner_path": str(owner_marker),
        "fingerprint": stable_fingerprint(dict(payload)),
    }
    anchor = _read_lock_anchor(transaction_lock)
    anchor["owner"] = {"phase": "initializing", "record": record}
    _write_lock_anchor(transaction_lock, anchor)
    _after_owner_marker_intent_boundary(initializing_marker)
    with initializing_marker.open("xb") as handle:
        record = {
            **record,
            "instance": _status_instance(os.fstat(handle.fileno())),
        }
        anchor = _read_lock_anchor(transaction_lock)
        anchor["owner"] = {"phase": "created", "record": record}
        _write_lock_anchor(transaction_lock, anchor)
        _after_owner_marker_created_anchor_boundary(initializing_marker)
        _write_all(handle, encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(initializing_marker.parent)
    _after_owner_marker_payload_boundary(initializing_marker)
    record = {
        "identity": _file_identity_payload(initializing_marker),
        "fingerprint": stable_fingerprint(dict(payload)),
        "initializing_path": str(initializing_marker),
        "owner_path": str(owner_marker),
    }
    anchor = _read_lock_anchor(transaction_lock)
    anchor["owner"] = {"phase": "transition", "record": record}
    _write_lock_anchor(transaction_lock, anchor)
    expected = _identity_from_payload(record["identity"])
    _move_entry_to_quarantine_exact(
        initializing_marker, owner_marker, expected, kind="file"
    )
    _after_owner_marker_replace_boundary(owner_marker)
    identity = file_identity(owner_marker)
    anchor = _read_lock_anchor(transaction_lock)
    anchor["owner"] = {"phase": "active", "record": record}
    _write_lock_anchor(transaction_lock, anchor)
    return identity


def _after_owner_marker_intent_boundary(_initializing_marker: Path) -> None:
    """Test seam after owner-marker creation intent is durable."""


def _after_owner_marker_created_anchor_boundary(_initializing_marker: Path) -> None:
    """Test seam after an empty owner-marker inode is identity-anchored."""


def _after_owner_marker_payload_boundary(_initializing_marker: Path) -> None:
    """Test seam after owner payload fsync and before the full identity anchor."""


def _after_owner_marker_replace_boundary(_owner_marker: Path) -> None:
    """Test seam after owner publication and before the building journal state."""


def _after_work_directory_creation_intent_boundary(_work: Path) -> None:
    """Test seam after the private work path intent is durable but before mkdir."""


def _load_shard_transaction(
    journal: Path,
    *,
    output: Path,
    request_fingerprint: str,
    dataset: str,
    transaction_lock: BinaryIO,
) -> tuple[dict[str, Any], FileIdentity]:
    if _journal_replacement_record(journal, transaction_lock) is not None:
        _complete_journal_replacement(journal, transaction_lock)
    before = file_identity(journal)
    payload = _read_json_object(journal, "build transaction")
    if file_identity(journal) != before:
        raise RuntimeError(f"shard build transaction changed while reading: {journal}")
    anchor = _read_lock_anchor(transaction_lock)
    journal_anchor = anchor.get("journal")
    accepted = (
        journal_anchor.get("accepted", []) if isinstance(journal_anchor, dict) else []
    )
    if not any(
        _record_matches_payload(journal, record, payload) for record in accepted
    ):
        raise RuntimeError(f"shard build transaction identity changed: {journal}")
    transaction_id = payload.get("transaction_id")
    if (
        payload.get("schema_version") != _SHARD_BUILD_TRANSACTION_SCHEMA
        or payload.get("state")
        not in {
            "initializing",
            "owner_initializing",
            "building",
            "publishing",
            "rolled_back",
        }
        or not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
    ):
        raise RuntimeError("shard build transaction contract is invalid")
    state = str(payload["state"])
    work_instance = payload.get("work_instance")
    owner_identity = payload.get("owner_identity")
    if state == "initializing" and (
        work_instance is not None or owner_identity is not None
    ):
        raise RuntimeError("initializing shard transaction claims artifact identity")
    if state == "owner_initializing" and (
        not isinstance(work_instance, list)
        or len(work_instance) != 3
        or any(type(value) is not int for value in work_instance)
        or owner_identity is not None
    ):
        raise RuntimeError("shard work initialization identity is invalid")
    if state in {"building", "publishing", "rolled_back"} and (
        not isinstance(work_instance, list)
        or len(work_instance) != 3
        or any(type(value) is not int for value in work_instance)
        or not isinstance(owner_identity, list)
        or len(owner_identity) != 5
        or any(type(value) is not int for value in owner_identity)
    ):
        raise RuntimeError("active shard transaction identity is invalid")
    expected_work = output / f".shards-{transaction_id}.partial"
    expected_owner = expected_work / _SHARD_WORK_OWNER_FILE
    expected_initializing = expected_work / _SHARD_WORK_OWNER_INITIALIZING_FILE
    expected_owner_payload = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "request_fingerprint": request_fingerprint,
    }
    if (
        payload.get("output_root") != str(output)
        or payload.get("request_fingerprint") != request_fingerprint
        or payload.get("dataset_root") != str(output / f"dataset={dataset}")
        or payload.get("work") != str(expected_work)
        or payload.get("owner_marker") != str(expected_owner)
        or payload.get("owner_initializing_marker") != str(expected_initializing)
        or payload.get("owner_payload") != expected_owner_payload
    ):
        raise RuntimeError(
            "shard build transaction does not match the requested output contract"
        )
    cleanup_inventory = payload.get("cleanup_inventory")
    if cleanup_inventory is not None and not isinstance(cleanup_inventory, list):
        raise RuntimeError("shard cleanup inventory contract is invalid")
    cleanup_removed = payload.get("cleanup_removed")
    cleanup_intent = payload.get("cleanup_removal_intent")
    payload.setdefault("cleanup_quarantine_intent", None)
    payload.setdefault("work_rollback_intent", None)
    cleanup_quarantine_intent = payload.get("cleanup_quarantine_intent")
    work_rollback_intent = payload.get("work_rollback_intent")
    if (
        not isinstance(cleanup_removed, list)
        or any(not isinstance(value, str) for value in cleanup_removed)
        or (cleanup_intent is not None and not isinstance(cleanup_intent, str))
        or (
            cleanup_inventory is None
            and (cleanup_removed or cleanup_intent is not None)
        )
        or (
            cleanup_quarantine_intent is not None
            and not isinstance(cleanup_quarantine_intent, dict)
        )
        or (
            work_rollback_intent is not None
            and not isinstance(work_rollback_intent, dict)
        )
    ):
        raise RuntimeError("shard cleanup progress contract is invalid")
    if state in {"publishing", "rolled_back"}:
        legacy_directory_progress = (
            "directory_rollback_removed" not in payload
            or "directory_rollback_intent" not in payload
        )
        payload.setdefault("directory_rollback_removed", [])
        payload.setdefault("directory_rollback_intent", None)
        rollback_removed = payload.get("rollback_removed")
        rollback_intent = payload.get("rollback_intent")
        directory_rollback_removed = payload.get("directory_rollback_removed")
        directory_rollback_intent = payload.get("directory_rollback_intent")
        if (
            not isinstance(payload.get("artifacts"), list)
            or type(payload.get("published_prefix")) is not int
            or int(payload["published_prefix"]) < 0
            or not isinstance(payload.get("published_directories"), list)
            or (
                payload.get("directory_intent") is not None
                and not isinstance(payload.get("directory_intent"), dict)
            )
            or not isinstance(rollback_removed, list)
            or any(not isinstance(value, str) for value in rollback_removed)
            or (rollback_intent is not None and not isinstance(rollback_intent, str))
            or not isinstance(directory_rollback_removed, list)
            or any(not isinstance(value, str) for value in directory_rollback_removed)
            or (
                directory_rollback_intent is not None
                and not isinstance(directory_rollback_intent, dict)
            )
        ):
            raise RuntimeError("shard publication transaction contract is invalid")
        if legacy_directory_progress:
            before = _replace_shard_transaction(
                journal, payload, before, transaction_lock
            )
    return payload, before


def _owner_identity_from_anchor(
    transaction: Mapping[str, Any], transaction_lock: BinaryIO
) -> list[int]:
    anchor = _read_lock_anchor(transaction_lock)
    owner_anchor = anchor.get("owner")
    if not isinstance(owner_anchor, dict) or owner_anchor.get("phase") not in {
        "transition",
        "active",
    }:
        raise RuntimeError("shard owner marker identity anchor is missing")
    record = owner_anchor.get("record")
    if not isinstance(record, dict):
        raise RuntimeError("shard owner marker identity anchor is invalid")
    owner = Path(str(transaction["owner_marker"]))
    initializing = Path(str(transaction["owner_initializing_marker"]))
    if record.get("owner_path") != str(owner) or record.get("initializing_path") != str(
        initializing
    ):
        raise RuntimeError("shard owner marker identity anchor path changed")
    candidates = [path for path in (owner, initializing) if _path_entry_exists(path)]
    if len(candidates) != 1 or not _record_matches_payload(
        candidates[0], record, cast(Mapping[str, Any], transaction["owner_payload"])
    ):
        raise RuntimeError(f"shard owner marker identity changed: {owner}")
    return cast(list[int], record["identity"])


def _after_owner_marker_retirement_intent_boundary(_marker: Path) -> None:
    """Test seam after the marker retirement identity is durable."""


def _before_owner_marker_retirement_quarantine_boundary(_marker: Path) -> None:
    """Test seam before atomically quarantining an intended owner marker."""


def _after_owner_marker_retirement_removal_boundary(_marker: Path) -> None:
    """Test seam after marker unlink and immediate-parent fsync."""


def _after_owner_marker_retirement_progress_boundary(_marker: Path) -> None:
    """Test seam after durable marker retirement progress is cleared."""


def _complete_owner_marker_retirement(
    transaction: Mapping[str, Any], transaction_lock: BinaryIO
) -> None:
    anchor = _read_lock_anchor(transaction_lock)
    owner_anchor = anchor.get("owner")
    if not isinstance(owner_anchor, dict) or owner_anchor.get("phase") != "retiring":
        raise RuntimeError("shard owner marker retirement anchor is invalid")
    record = owner_anchor.get("record")
    if not isinstance(record, dict):
        raise RuntimeError("shard owner marker retirement anchor is invalid")
    owner = Path(str(transaction["owner_marker"]))
    initializing = Path(str(transaction["owner_initializing_marker"]))
    output = Path(str(transaction["output_root"]))
    marker = Path(str(record.get("retiring_path", "")))
    quarantine = Path(str(record.get("quarantine_path", "")))
    if (
        record.get("owner_path") != str(owner)
        or record.get("initializing_path") != str(initializing)
        or marker not in {owner, initializing}
        or not _retirement_path_matches(output, quarantine, kind="file")
        or record.get("retirement_phase")
        not in {"intended", "quarantined", "truncate_intended"}
        or not isinstance(record.get("identity"), list)
        or len(cast(list[object], record["identity"])) != 5
        or any(
            type(value) is not int for value in cast(list[object], record["identity"])
        )
        or type(record.get("allow_empty", False)) is not bool
    ):
        raise RuntimeError("shard owner marker retirement anchor is invalid")
    retirement_record = cast(dict[str, object], record)

    def validate_quarantine() -> bool:
        if not _path_entry_exists(quarantine):
            return False
        truncate_intended = (
            retirement_record.get("retirement_phase") == "truncate_intended"
        )
        identity_matches = _identity_payload_matches(
            quarantine, retirement_record["identity"]
        )
        stable_identity_matches = (
            isinstance(retirement_record.get("identity"), list)
            and _entry_instance(quarantine)
            == cast(list[int], retirement_record["identity"])[:3]
            and file_identity(quarantine).size == 0
        )
        if (
            _path_is_link_like(quarantine)
            or not _is_single_link_regular_file(quarantine)
            or not (identity_matches or (truncate_intended and stable_identity_matches))
        ):
            raise RuntimeError(f"shard owner marker identity changed: {quarantine}")
        if truncate_intended and stable_identity_matches:
            return True
        if bool(retirement_record.get("allow_empty", False)):
            if file_identity(quarantine).size != 0:
                raise RuntimeError(f"shard owner marker identity changed: {quarantine}")
        else:
            try:
                observed = _read_json_object(quarantine, "owner marker")
            except RuntimeError:
                raise RuntimeError(
                    f"shard owner marker identity changed: {quarantine}"
                ) from None
            if observed != transaction["owner_payload"] or retirement_record.get(
                "fingerprint"
            ) != stable_fingerprint(observed):
                raise RuntimeError(
                    f"foreign shard owner marker blocks cleanup: {quarantine}"
                )
        return True

    if record["retirement_phase"] == "intended":
        if _path_entry_exists(marker):
            if _path_entry_exists(quarantine):
                raise RuntimeError(
                    f"shard owner marker quarantine already exists: {quarantine}"
                )
            _before_owner_marker_retirement_quarantine_boundary(marker)
            try:
                _move_entry_to_quarantine_exact(
                    marker,
                    quarantine,
                    _identity_from_payload(record["identity"]),
                    kind="file",
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"shard owner marker identity changed: {marker}"
                ) from exc
        elif not _path_entry_exists(quarantine):
            raise RuntimeError(
                f"shard owner marker disappeared without retirement progress: {marker}"
            )
        try:
            if not validate_quarantine():
                raise RuntimeError(
                    f"shard owner marker quarantine disappeared: {quarantine}"
                )
        except RuntimeError:
            if not _path_entry_exists(marker) and _path_entry_exists(quarantine):
                try:
                    _rename_directory_no_replace(quarantine, marker)
                    _fsync_directory(marker.parent)
                    if quarantine.parent != marker.parent:
                        _fsync_directory(quarantine.parent)
                except BaseException:
                    pass
            raise
        record = {**record, "retirement_phase": "quarantined"}
        anchor = _read_lock_anchor(transaction_lock)
        anchor["owner"] = {"phase": "retiring", "record": record}
        _write_lock_anchor(transaction_lock, anchor)
    if validate_quarantine():
        if record[
            "retirement_phase"
        ] == "quarantined" and not _identity_payload_matches(
            quarantine, record["identity"]
        ):
            raise RuntimeError(f"shard owner marker identity changed: {quarantine}")
        expected = _identity_from_payload(record["identity"])
        if os.name == "nt" and not _uses_private_trash_retirement():
            _unlink_file_if_identity(quarantine, expected)
        else:
            if record["retirement_phase"] == "quarantined":
                record = {**record, "retirement_phase": "truncate_intended"}
                anchor = _read_lock_anchor(transaction_lock)
                anchor["owner"] = {"phase": "retiring", "record": record}
                _write_lock_anchor(transaction_lock, anchor)
            _retire_regular_file_without_pathname_delete(
                quarantine, expected, transaction_lock
            )
        _fsync_directory(quarantine.parent)
        _after_owner_marker_retirement_removal_boundary(marker)
    anchor = _read_lock_anchor(transaction_lock)
    current = anchor.get("owner")
    if (
        not isinstance(current, dict)
        or current.get("phase") != "retiring"
        or current.get("record") != record
    ):
        raise RuntimeError("shard owner marker retirement anchor changed")
    anchor["owner"] = None
    _write_lock_anchor(transaction_lock, anchor)
    _after_owner_marker_retirement_progress_boundary(marker)


def _begin_owner_marker_retirement(
    transaction: Mapping[str, Any],
    transaction_lock: BinaryIO,
    *,
    marker: Path,
    record: Mapping[str, Any],
    allow_empty: bool = False,
) -> None:
    if (
        _path_is_link_like(marker)
        or not _is_single_link_regular_file(marker)
        or not _identity_payload_matches(marker, record.get("identity"))
    ):
        raise RuntimeError(f"shard owner marker identity changed: {marker}")
    retiring_record = {
        **dict(record),
        "identity": _file_identity_payload(marker),
        "retiring_path": str(marker),
        "quarantine_path": str(
            _retired_entry_path(Path(str(transaction["output_root"])), "file")
        ),
        "retirement_phase": "intended",
        "allow_empty": allow_empty,
    }
    anchor = _read_lock_anchor(transaction_lock)
    anchor["owner"] = {"phase": "retiring", "record": retiring_record}
    _write_lock_anchor(transaction_lock, anchor)
    _after_owner_marker_retirement_intent_boundary(marker)
    _complete_owner_marker_retirement(transaction, transaction_lock)


def _normalize_incomplete_owner_anchor(
    transaction: Mapping[str, Any], transaction_lock: BinaryIO
) -> bool:
    anchor = _read_lock_anchor(transaction_lock)
    owner_anchor = anchor.get("owner")
    if not isinstance(owner_anchor, dict):
        if owner_anchor is None:
            return False
        raise RuntimeError("shard owner marker identity anchor is invalid")
    phase = owner_anchor.get("phase")
    if phase in {"transition", "active"}:
        return True
    if phase == "retiring":
        _complete_owner_marker_retirement(transaction, transaction_lock)
        return False
    record = owner_anchor.get("record")
    if not isinstance(record, dict):
        raise RuntimeError("shard owner marker identity anchor is invalid")
    owner = Path(str(transaction["owner_marker"]))
    initializing = Path(str(transaction["owner_initializing_marker"]))
    if (
        record.get("owner_path") != str(owner)
        or record.get("initializing_path") != str(initializing)
        or _path_entry_exists(owner)
    ):
        raise RuntimeError(f"shard owner marker identity changed: {owner}")
    if phase == "initializing":
        if _path_entry_exists(initializing):
            raise RuntimeError(f"shard owner marker identity changed: {owner}")
    elif phase == "created":
        expected = record.get("instance")
        if not _path_entry_exists(initializing):
            raise RuntimeError(
                f"shard owner marker disappeared without retirement progress: {initializing}"
            )
        if _path_entry_exists(initializing):
            if (
                _path_is_link_like(initializing)
                or not _is_single_link_regular_file(initializing)
                or _entry_instance(initializing) != expected
            ):
                raise RuntimeError(f"shard owner marker identity changed: {owner}")
            if file_identity(initializing).size:
                try:
                    observed = _read_json_object(initializing, "owner temporary")
                except RuntimeError:
                    raise RuntimeError(
                        f"shard owner marker identity changed: {owner}"
                    ) from None
                if record.get("fingerprint") != stable_fingerprint(observed):
                    raise RuntimeError(f"shard owner marker identity changed: {owner}")
            if _entry_instance(initializing) != expected:
                raise RuntimeError(f"shard owner marker identity changed: {owner}")
            created_record = {
                **record,
                "identity": _file_identity_payload(initializing),
            }
            _begin_owner_marker_retirement(
                transaction,
                transaction_lock,
                marker=initializing,
                record=created_record,
                allow_empty=file_identity(initializing).size == 0,
            )
            return False
    else:
        raise RuntimeError("shard owner marker identity anchor phase is invalid")
    anchor = _read_lock_anchor(transaction_lock)
    anchor["owner"] = None
    _write_lock_anchor(transaction_lock, anchor)
    _after_owner_marker_retirement_progress_boundary(initializing)
    return False


def _validate_work_inventory_path(work: Path, path: Path, dataset: str) -> None:
    relative = path.relative_to(work)
    value = relative.as_posix()
    parts = relative.parts
    if value in {_SHARD_WORK_OWNER_FILE, _SHARD_WORK_OWNER_INITIALIZING_FILE}:
        return
    if value == "membership.snapshot.parquet":
        return
    if not parts:
        raise RuntimeError("shard cleanup inventory contains the work root")
    top = parts[0]
    run_patterns = {
        "source-runs": r"run-\d{8}\.parquet",
        "source-merges": r"source-merge-\d{4}-\d{6}\.parquet",
        "joined-runs": r"run-\d{8}\.parquet",
        "joined-merges": r"joined-merge-\d{4}-\d{6}\.parquet",
    }
    if top in run_patterns:
        if len(parts) == 1 and path.is_dir():
            return
        if len(parts) == 2 and re.fullmatch(run_patterns[top], parts[1]):
            return
    if top == "staged":
        expected = (
            f"dataset={dataset}",
            "split=",
            "label=",
        )
        if len(parts) == 1 and path.is_dir():
            return
        if len(parts) >= 2 and parts[1] == expected[0]:
            if len(parts) == 2 and path.is_dir():
                return
            if len(parts) >= 3 and parts[2] in {
                "split=train",
                "split=validation",
                "split=test",
            }:
                if len(parts) == 3 and path.is_dir():
                    return
                if len(parts) >= 4 and re.fullmatch(
                    r"label=[a-z0-9][a-z0-9_]*", parts[3]
                ):
                    if len(parts) == 4 and path.is_dir():
                        return
                    if len(parts) == 5 and re.fullmatch(
                        r"(?:part-\d{8}\.parquet|\.part-\d{8}\.parquet\.partial)",
                        parts[4],
                    ):
                        return
    raise RuntimeError(f"foreign path blocks shard cleanup inventory: {path}")


def _capture_owned_work_inventory(
    transaction: Mapping[str, Any],
) -> list[dict[str, Any]]:
    work = Path(str(transaction["work"]))
    dataset = Path(str(transaction["dataset_root"])).name.removeprefix("dataset=")
    inventory: list[dict[str, Any]] = []
    for path in sorted(
        work.rglob("*"), key=lambda value: value.relative_to(work).as_posix()
    ):
        if _path_is_link_like(path):
            raise RuntimeError(f"linked path blocks shard cleanup inventory: {path}")
        _validate_work_inventory_path(work, path, dataset)
        relative = path.relative_to(work).as_posix()
        if path.is_dir():
            identity: list[int] = cast(list[int], _entry_instance(path))
            kind = "directory"
        elif path.is_file():
            identity = _file_identity_payload(path)
            kind = "file"
        else:
            raise RuntimeError(f"foreign entry blocks shard cleanup inventory: {path}")
        entry: dict[str, Any] = {
            "path": relative,
            "kind": kind,
            "identity": identity,
        }
        if kind == "file":
            entry["sha256"] = _sha256_file(path)
        inventory.append(entry)
    return inventory


def _ordered_cleanup_inventory(
    work: Path, inventory: Sequence[object]
) -> list[tuple[Path, Mapping[str, Any]]]:
    parsed: list[tuple[Path, Mapping[str, Any]]] = []
    for raw in inventory:
        if not isinstance(raw, dict):
            raise RuntimeError("shard cleanup inventory entry is invalid")
        entry = cast(Mapping[str, Any], raw)
        parsed.append((_owned_inventory_path(work, entry.get("path")), entry))
    parsed.sort(
        key=lambda item: (len(item[0].relative_to(work).parts), str(item[0])),
        reverse=True,
    )
    return parsed


def _validate_persisted_quarantine_intent(
    raw: object,
    *,
    output: Path,
    source: Path,
    kind: str,
    identity: object,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("shard cleanup quarantine progress is invalid")
    quarantine = Path(str(raw.get("quarantine_path", "")))
    if (
        raw.get("source_path") != str(source)
        or raw.get("kind") != kind
        or raw.get("identity") != identity
        or raw.get("phase") not in {"intended", "quarantined", "gone", "debt_recorded"}
        or not _retirement_path_matches(output, quarantine, kind=kind)
    ):
        raise RuntimeError("shard cleanup quarantine progress is invalid")
    return cast(dict[str, Any], raw)


def _validate_cleanup_progress(transaction: Mapping[str, Any]) -> None:
    work = Path(str(transaction["work"]))
    inventory = transaction.get("cleanup_inventory")
    removed = transaction.get("cleanup_removed")
    intent = transaction.get("cleanup_removal_intent")
    quarantine_intent = transaction.get("cleanup_quarantine_intent")
    if not isinstance(inventory, list) or not isinstance(removed, list):
        raise RuntimeError("shard cleanup progress was not durably recorded")
    ordered = _ordered_cleanup_inventory(work, inventory)
    ordered_paths = [path.relative_to(work).as_posix() for path, _ in ordered]
    if removed != ordered_paths[: len(removed)] or len(removed) > len(ordered_paths):
        raise RuntimeError("shard cleanup removal progress is invalid")
    next_path = (
        ordered_paths[len(removed)] if len(removed) < len(ordered_paths) else None
    )
    if intent is not None and intent != next_path:
        raise RuntimeError("shard cleanup removal intent is invalid")
    if (intent is None) != (quarantine_intent is None):
        raise RuntimeError("shard cleanup quarantine progress is invalid")
    if intent is not None:
        entry = dict(ordered[len(removed)][1])
        _validate_persisted_quarantine_intent(
            quarantine_intent,
            output=Path(str(transaction["output_root"])),
            source=_owned_inventory_path(work, intent),
            kind=str(entry.get("kind")),
            identity=entry.get("identity"),
        )
    expected = {str(entry["path"]): dict(entry) for _, entry in ordered[len(removed) :]}
    observed = {
        str(entry["path"]): entry
        for entry in _capture_owned_work_inventory(transaction)
    }
    if intent is not None and intent not in observed:
        expected.pop(intent, None)
    if observed != expected:
        raise RuntimeError("shard cleanup inventory changed")


def _prepare_owned_shard_work_cleanup(
    transaction: dict[str, Any],
    journal: Path,
    journal_identity: FileIdentity,
    transaction_lock: BinaryIO,
) -> FileIdentity:
    work = Path(str(transaction["work"]))
    root_intent = transaction.get("work_rollback_intent")
    if not _path_entry_exists(work):
        if root_intent is None:
            return journal_identity
        if not isinstance(root_intent, dict):
            raise RuntimeError("shard work rollback progress is invalid")
        identity = root_intent.get("identity")
        if not isinstance(identity, list) or len(identity) != 3:
            raise RuntimeError("shard work rollback identity is invalid")
        journal_identity = _finish_persisted_transaction_quarantine(
            transaction,
            intent_field="work_rollback_intent",
            source=work,
            expected=cast(list[int], identity),
            kind="directory",
            journal=journal,
            journal_identity=journal_identity,
            transaction_lock=transaction_lock,
            before_quarantine=_before_owned_work_root_quarantine_boundary,
        )
        transaction["work_rollback_intent"] = None
        return _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
    if transaction["state"] not in {"building", "publishing"}:
        return journal_identity
    owner = Path(str(transaction["owner_marker"]))
    initializing = Path(str(transaction["owner_initializing_marker"]))
    if _path_is_link_like(work) or not work.is_dir():
        raise RuntimeError(f"foreign shard work path blocks cleanup: {work}")
    if _entry_instance(work) != transaction.get("work_instance"):
        raise RuntimeError(f"shard work directory identity changed: {work}")
    expected = transaction.get("cleanup_inventory")
    if expected is not None:
        _validate_cleanup_progress(transaction)
        return journal_identity
    if _path_entry_exists(initializing):
        raise RuntimeError(
            f"ambiguous shard owner initialization blocks cleanup: {initializing}"
        )
    if (
        _path_is_link_like(owner)
        or not owner.is_file()
        or not _identity_payload_matches(owner, transaction.get("owner_identity"))
    ):
        raise RuntimeError(f"shard owner marker identity changed: {owner}")
    try:
        observed_owner = json.loads(owner.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"shard owner marker cannot be validated: {owner}") from exc
    if observed_owner != transaction["owner_payload"]:
        raise RuntimeError(f"foreign shard owner marker blocks cleanup: {owner}")
    current = _capture_owned_work_inventory(transaction)
    transaction["cleanup_inventory"] = current
    transaction["cleanup_removed"] = []
    transaction["cleanup_removal_intent"] = None
    return _replace_shard_transaction(
        journal, transaction, journal_identity, transaction_lock
    )


def _before_owned_shard_work_cleanup(_work: Path) -> None:
    """Test seam immediately before identity-aware owned work cleanup."""


def _before_owned_inventory_entry_removal(_path: Path) -> None:
    """Test seam before the final identity/content check for one owned entry."""


def _after_owned_inventory_entry_removal(_path: Path) -> None:
    """Test seam after an owned entry is durably removed but before progress advances."""


def _before_owned_work_root_removal(_work: Path) -> None:
    """Test seam after owned entries are removed but before empty-root removal."""


def _before_owned_inventory_quarantine_boundary(_path: Path) -> None:
    """Test seam at the final namespace syscall for one inventory entry."""


def _before_owned_work_root_quarantine_boundary(_work: Path) -> None:
    """Test seam at the final namespace syscall for the owned work root."""


def _finish_persisted_transaction_quarantine(
    transaction: dict[str, Any],
    *,
    intent_field: str,
    source: Path,
    expected: FileIdentity | Sequence[int],
    kind: str,
    journal: Path,
    journal_identity: FileIdentity,
    transaction_lock: BinaryIO,
    before_quarantine: Callable[[Path], None],
) -> FileIdentity:
    output = Path(str(transaction["output_root"]))
    identity = (
        _identity_payload(expected)
        if isinstance(expected, FileIdentity)
        else list(expected)
    )
    intent = _validate_persisted_quarantine_intent(
        transaction.get(intent_field),
        output=output,
        source=source,
        kind=kind,
        identity=identity,
    )
    quarantine = Path(str(intent["quarantine_path"]))
    phase = str(intent["phase"])
    if phase == "intended":
        if _path_entry_exists(source):
            if not _entry_matches_expected(source, expected, kind=kind):
                raise RuntimeError(f"shard cleanup {kind} identity changed: {source}")
            if _path_entry_exists(quarantine):
                raise RuntimeError(
                    f"shard cleanup quarantine already exists: {quarantine}"
                )
            before_quarantine(source)
            if kind == "directory" and any(source.iterdir()):
                raise OSError(
                    errno.ENOTEMPTY,
                    "owned shard directory became non-empty before quarantine",
                    str(source),
                )
            _move_entry_to_quarantine_exact(source, quarantine, expected, kind=kind)
        elif not _entry_matches_expected(quarantine, expected, kind=kind):
            if not _recorded_retired_entry(transaction_lock, quarantine, kind=kind):
                raise RuntimeError(
                    f"shard cleanup {kind} disappeared without quarantine: {source}"
                )
        intent = {**intent, "phase": "quarantined"}
        transaction[intent_field] = intent
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
        phase = "quarantined"
    if phase == "quarantined":
        if _path_entry_exists(source):
            raise RuntimeError(f"foreign {kind} blocks shard cleanup: {source}")
        outcome = _retire_quarantined_entry(
            quarantine,
            expected,
            kind=kind,
            transaction_lock=transaction_lock,
        )
        intent = {**intent, "phase": outcome}
        transaction[intent_field] = intent
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
        phase = outcome
    if phase == "gone":
        if not (
            os.name == "nt" or _uses_private_trash_retirement()
        ) or _path_entry_exists(quarantine):
            raise RuntimeError(
                f"shard cleanup quarantine identity changed: {quarantine}"
            )
    elif phase == "debt_recorded":
        if not _recorded_retired_entry(transaction_lock, quarantine, kind=kind):
            raise RuntimeError(
                f"shard cleanup quarantine identity changed: {quarantine}"
            )
    else:
        raise RuntimeError("shard cleanup quarantine phase is invalid")
    return journal_identity


def _owned_inventory_path(work: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("shard cleanup inventory path is invalid")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RuntimeError("shard cleanup inventory path is invalid")
    path = work.joinpath(*relative.parts)
    if path.parent != work and work not in path.parents:
        raise RuntimeError("shard cleanup inventory path escaped work root")
    return path


def _remove_owned_inventory_entries(
    work: Path,
    transaction: dict[str, Any],
    journal: Path,
    journal_identity: FileIdentity,
    transaction_lock: BinaryIO,
) -> FileIdentity:
    inventory = cast(Sequence[object], transaction["cleanup_inventory"])
    parsed = _ordered_cleanup_inventory(work, inventory)
    removed = cast(list[str], transaction["cleanup_removed"])
    for path, entry in parsed[len(removed) :]:
        relative = path.relative_to(work).as_posix()
        intent = transaction.get("cleanup_removal_intent")
        if intent is None:
            transaction["cleanup_removal_intent"] = relative
            transaction["cleanup_quarantine_intent"] = {
                "source_path": str(path),
                "quarantine_path": str(
                    _retired_entry_path(
                        Path(str(transaction["output_root"])), str(entry.get("kind"))
                    )
                ),
                "kind": entry.get("kind"),
                "identity": entry.get("identity"),
                "phase": "intended",
            }
            journal_identity = _replace_shard_transaction(
                journal, transaction, journal_identity, transaction_lock
            )
        elif intent != relative:
            raise RuntimeError("shard cleanup removal intent is invalid")
        kind = entry.get("kind")
        identity = entry.get("identity")
        expected: FileIdentity | list[int]
        if kind == "file":
            expected = _identity_from_payload(identity)
            expected_digest = entry.get("sha256")
            if _path_entry_exists(path):
                _before_owned_inventory_entry_removal(path)
                if (
                    not isinstance(expected_digest, str)
                    or len(expected_digest) != 64
                    or _path_is_link_like(path)
                    or not path.is_file()
                    or file_identity(path) != expected
                    or _sha256_file(path) != expected_digest
                    or file_identity(path) != expected
                ):
                    raise RuntimeError(
                        f"shard cleanup inventory file identity changed: {path}"
                    )
        elif kind == "directory":
            if (
                not isinstance(identity, list)
                or len(identity) != 3
                or any(type(value) is not int for value in identity)
            ):
                raise RuntimeError(
                    "shard cleanup inventory directory identity is invalid"
                )
            expected = cast(list[int], identity)
            if _path_entry_exists(path):
                _before_owned_inventory_entry_removal(path)
                if (
                    _path_is_link_like(path)
                    or not path.is_dir()
                    or _entry_instance(path) != identity
                    or any(path.iterdir())
                ):
                    raise RuntimeError(
                        f"shard cleanup inventory directory identity changed: {path}"
                    )
        else:
            raise RuntimeError("shard cleanup inventory kind is invalid")
        journal_identity = _finish_persisted_transaction_quarantine(
            transaction,
            intent_field="cleanup_quarantine_intent",
            source=path,
            expected=expected,
            kind=cast(str, kind),
            journal=journal,
            journal_identity=journal_identity,
            transaction_lock=transaction_lock,
            before_quarantine=_before_owned_inventory_quarantine_boundary,
        )
        _after_owned_inventory_entry_removal(path)
        removed.append(relative)
        transaction["cleanup_removal_intent"] = None
        transaction["cleanup_quarantine_intent"] = None
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
    return journal_identity


def _cleanup_owned_shard_work(
    transaction: dict[str, Any],
    journal: Path,
    journal_identity: FileIdentity,
    transaction_lock: BinaryIO,
) -> FileIdentity:
    work = Path(str(transaction["work"]))
    if not _path_entry_exists(work):
        return journal_identity
    if _is_link_like(work) or not work.is_dir():
        raise RuntimeError(f"foreign shard work path blocks cleanup: {work}")
    state = str(transaction["state"])
    expected_work_instance = transaction.get("work_instance")
    entries = list(work.iterdir())
    owner = Path(str(transaction["owner_marker"]))
    initializing = Path(str(transaction["owner_initializing_marker"]))
    if state == "initializing":
        raise RuntimeError(
            f"ambiguous shard ownership initialization blocks cleanup: {work}"
        )
    else:
        if _entry_instance(work) != expected_work_instance:
            raise RuntimeError(f"shard work directory identity changed: {work}")
        if state == "owner_initializing":
            owner_materialized = _normalize_incomplete_owner_anchor(
                transaction, transaction_lock
            )
            anchored_owner_identity = (
                _owner_identity_from_anchor(transaction, transaction_lock)
                if owner_materialized
                else []
            )
            entries = list(work.iterdir())
            unexpected = [path for path in entries if path not in {owner, initializing}]
            if unexpected:
                raise RuntimeError(
                    f"foreign path blocks shard ownership initialization cleanup: {unexpected[0]}"
                )
            if not owner_materialized:
                if entries:
                    raise RuntimeError(
                        f"foreign path blocks shard ownership initialization cleanup: {entries[0]}"
                    )
            elif _path_entry_exists(owner):
                if _is_link_like(owner) or not owner.is_file():
                    raise RuntimeError(
                        f"foreign shard owner marker blocks cleanup: {owner}"
                    )
                try:
                    observed_owner = json.loads(owner.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"shard owner marker cannot be validated: {owner}"
                    ) from exc
                if observed_owner != transaction["owner_payload"]:
                    raise RuntimeError(
                        f"foreign shard owner marker blocks cleanup: {owner}"
                    )
                if not _identity_payload_matches(owner, anchored_owner_identity):
                    raise RuntimeError(f"shard owner marker identity changed: {owner}")
            elif not _identity_payload_matches(initializing, anchored_owner_identity):
                raise RuntimeError(f"shard owner marker identity changed: {owner}")
        else:
            expected_inventory = transaction.get("cleanup_inventory")
            if not isinstance(expected_inventory, list):
                raise RuntimeError("shard cleanup inventory was not durably recorded")
            _validate_cleanup_progress(transaction)
    _before_owned_shard_work_cleanup(work)
    if state != "initializing" and _entry_instance(work) != expected_work_instance:
        raise RuntimeError(
            f"shard work directory identity changed before cleanup: {work}"
        )
    if state in {"building", "publishing"}:
        _validate_cleanup_progress(transaction)
    output = Path(str(transaction["output_root"])).resolve(strict=True)
    resolved_work = work.resolve(strict=True)
    if (
        resolved_work.parent != output
        or resolved_work.name != f".shards-{transaction['transaction_id']}.partial"
    ):
        raise RuntimeError(f"owned shard work target escaped output root: {work}")
    if state in {"building", "publishing"}:
        inventory = transaction.get("cleanup_inventory")
        if not isinstance(inventory, list):
            raise RuntimeError("shard cleanup inventory was not durably recorded")
        journal_identity = _remove_owned_inventory_entries(
            work, transaction, journal, journal_identity, transaction_lock
        )
    elif state == "owner_initializing":
        remaining = list(work.iterdir())
        unexpected = [path for path in remaining if path not in {owner, initializing}]
        if unexpected:
            raise RuntimeError(
                f"foreign path blocks shard ownership initialization cleanup: {unexpected[0]}"
            )
        if remaining:
            if len(remaining) != 1:
                raise RuntimeError(
                    "ambiguous shard ownership initialization cleanup entries"
                )
            path = remaining[0]
            anchor = _read_lock_anchor(transaction_lock)
            owner_anchor = anchor.get("owner")
            record = (
                owner_anchor.get("record") if isinstance(owner_anchor, dict) else None
            )
            expected = record.get("identity") if isinstance(record, dict) else None
            if not _identity_payload_matches(path, expected):
                raise RuntimeError(f"shard owner marker identity changed: {path}")
            _before_owned_inventory_entry_removal(path)
            if not _identity_payload_matches(path, expected):
                raise RuntimeError(f"shard owner marker identity changed: {path}")
            _begin_owner_marker_retirement(
                transaction,
                transaction_lock,
                marker=path,
                record=cast(Mapping[str, Any], record),
            )
        if list(work.iterdir()):
            raise RuntimeError(
                f"foreign path blocks shard ownership initialization cleanup: {work}"
            )
    elif list(work.iterdir()):
        raise RuntimeError(f"foreign path blocks shard ownership cleanup: {work}")
    root_instance = _entry_instance(work)
    if root_instance is None:
        raise RuntimeError(f"shard work directory identity changed: {work}")
    _before_owned_work_root_removal(work)
    if _entry_instance(work) != root_instance:
        raise RuntimeError(
            f"shard work directory identity changed before removal: {work}"
        )
    if transaction.get("work_rollback_intent") is None:
        transaction["work_rollback_intent"] = {
            "source_path": str(work),
            "quarantine_path": str(
                _retired_entry_path(Path(str(transaction["output_root"])), "directory")
            ),
            "kind": "directory",
            "identity": root_instance,
            "phase": "intended",
        }
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
    journal_identity = _finish_persisted_transaction_quarantine(
        transaction,
        intent_field="work_rollback_intent",
        source=work,
        expected=root_instance,
        kind="directory",
        journal=journal,
        journal_identity=journal_identity,
        transaction_lock=transaction_lock,
        before_quarantine=_before_owned_work_root_quarantine_boundary,
    )
    transaction["work_rollback_intent"] = None
    return _replace_shard_transaction(
        journal, transaction, journal_identity, transaction_lock
    )


def _before_journal_retirement_quarantine_boundary(_journal: Path) -> None:
    """Test seam at the final namespace syscall for journal retirement."""


def _migrate_legacy_manifest_commit_anchor(
    payload: Mapping[str, Any],
    transaction_lock: BinaryIO,
    anchor: dict[str, Any],
) -> dict[str, Any]:
    output = Path(str(payload.get("output_root", "")))
    manifest = output / "shard_manifest.json"
    if _path_is_link_like(manifest) or not manifest.is_file():
        raise RuntimeError(f"manifest commit evidence is missing: {manifest}")
    try:
        manifest_payload = load_shard_manifest(manifest)
    except RuntimeError:
        raise RuntimeError(f"manifest commit identity changed: {manifest}") from None
    record = _anchored_file_record(manifest, manifest_payload)
    if not _record_matches_payload(manifest, record, manifest_payload):
        raise RuntimeError(f"manifest commit identity changed: {manifest}")
    migrated = {
        "phase": "published",
        "path": str(manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.partial")),
        "destination": str(manifest),
        "fingerprint": record["fingerprint"],
        "identity": record["identity"],
    }
    anchor["manifest"] = migrated
    _write_lock_anchor(transaction_lock, anchor)
    return migrated


def _validate_manifest_commit(
    payload: Mapping[str, Any], transaction_lock: BinaryIO
) -> None:
    anchor = _read_lock_anchor(transaction_lock)
    raw = anchor.get("manifest")
    if raw is None:
        if payload.get("state") != "publishing":
            return
        raw = _migrate_legacy_manifest_commit_anchor(payload, transaction_lock, anchor)
    if not isinstance(raw, dict) or raw.get("phase") not in {"published", "committed"}:
        raise RuntimeError("manifest commit anchor is invalid")
    manifest = Path(str(raw.get("destination", "")))
    try:
        manifest_payload = _read_json_object(manifest, "manifest")
    except RuntimeError:
        raise RuntimeError(f"manifest commit identity changed: {manifest}") from None
    if not _record_matches_payload(manifest, raw, manifest_payload):
        raise RuntimeError(f"manifest commit identity changed: {manifest}")
    artifacts = payload.get("artifacts")
    if (
        payload.get("state") != "publishing"
        or not isinstance(artifacts, list)
        or payload.get("published_prefix") != len(artifacts)
    ):
        raise RuntimeError("manifest commit lacks a complete publication journal")
    entries = manifest_payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(artifacts):
        raise RuntimeError("manifest commit artifact contract is invalid")
    manifest_contract: list[tuple[str, int, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("manifest commit artifact contract is invalid")
        manifest_contract.append(
            (
                PurePosixPath(str(entry.get("path"))).as_posix(),
                int(entry.get("byte_size", -1)),
                str(entry.get("sha256")),
            )
        )
    journal_contract: list[tuple[str, int, str]] = []
    output = Path(str(payload.get("output_root", "")))
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, dict):
            raise RuntimeError("manifest commit artifact contract is invalid")
        relative = PurePosixPath(str(raw_artifact.get("relative"))).as_posix()
        final = output.joinpath(*_canonical_entry_parts(relative))
        identity = _identity_from_payload(raw_artifact.get("identity"))
        size = raw_artifact.get("size")
        digest = raw_artifact.get("sha256")
        if (
            type(size) is not int
            or not isinstance(digest, str)
            or not _entry_matches_expected(final, identity, kind="file")
            or final.stat().st_size != size
            or _sha256_file(final) != digest
            or file_identity(final) != identity
        ):
            raise RuntimeError(f"manifest commit artifact changed: {final}")
        journal_contract.append((relative, size, digest))
    if sorted(manifest_contract) != sorted(journal_contract):
        raise RuntimeError("manifest commit artifact contract is invalid")
    if raw.get("phase") == "published":
        raw = {**raw, "phase": "committed"}
        anchor["manifest"] = raw
        _write_lock_anchor(transaction_lock, anchor)


def _complete_journal_retirement(journal: Path, transaction_lock: BinaryIO) -> bool:
    anchor = _read_lock_anchor(transaction_lock, allow_unrecorded_retirement=True)
    journal_anchor = anchor.get("journal")
    if (
        not isinstance(journal_anchor, dict)
        or journal_anchor.get("phase") != "retiring"
    ):
        return False
    accepted = journal_anchor.get("accepted")
    retirement = journal_anchor.get("retirement")
    if (
        not isinstance(accepted, list)
        or len(accepted) != 1
        or not isinstance(accepted[0], dict)
        or not isinstance(retirement, dict)
    ):
        raise RuntimeError("shard journal retirement anchor is invalid")
    record = cast(dict[str, Any], accepted[0])
    expected = _identity_from_payload(record.get("identity"))
    quarantine = Path(str(retirement.get("quarantine_path", "")))
    phase = retirement.get("phase")
    if (
        retirement.get("journal_path") != str(journal)
        or retirement.get("identity") != record.get("identity")
        or not _retirement_path_matches(journal.parent, quarantine, kind="file")
        or phase not in {"intended", "quarantined", "gone", "debt_recorded"}
    ):
        raise RuntimeError("shard journal retirement anchor is invalid")
    committed_payload = retirement.get("payload")
    if not isinstance(committed_payload, dict):
        candidate = (
            journal
            if _path_entry_exists(journal)
            else quarantine if _path_entry_exists(quarantine) else None
        )
        if candidate is None:
            raise RuntimeError("shard journal retirement payload evidence is missing")
        committed_payload = _read_json_object(candidate, "build transaction")
        if not _record_matches_payload(candidate, record, committed_payload):
            raise RuntimeError(f"shard build transaction identity changed: {candidate}")
        retirement = {
            **retirement,
            "payload": committed_payload,
            "payload_fingerprint": stable_fingerprint(committed_payload),
        }
        journal_anchor["retirement"] = retirement
        anchor["journal"] = journal_anchor
        _write_lock_anchor(transaction_lock, anchor)
    if retirement.get("payload_fingerprint") != stable_fingerprint(
        committed_payload
    ) or record.get("fingerprint") != stable_fingerprint(committed_payload):
        raise RuntimeError("shard journal retirement payload evidence is invalid")
    # A committed manifest is only a state marker.  Re-prove every final shard
    # identity, size, and digest before any retirement progress is accepted.
    _validate_manifest_commit(committed_payload, transaction_lock)
    if phase == "intended":
        if _path_entry_exists(journal):
            current = _read_json_object(journal, "build transaction")
            if current != committed_payload or not _record_matches_payload(
                journal, record, current
            ):
                raise RuntimeError(
                    f"shard build transaction identity changed: {journal}"
                )
            if _path_entry_exists(quarantine):
                raise RuntimeError(
                    f"shard journal quarantine already exists: {quarantine}"
                )
            _before_journal_retirement_quarantine_boundary(journal)
            _move_entry_to_quarantine_exact(journal, quarantine, expected, kind="file")
        elif not _entry_matches_expected(quarantine, expected, kind="file"):
            if not _recorded_retired_entry(transaction_lock, quarantine, kind="file"):
                raise RuntimeError(
                    f"shard build transaction disappeared without quarantine: {journal}"
                )
        retirement = {**retirement, "phase": "quarantined"}
        journal_anchor["retirement"] = retirement
        anchor = _read_lock_anchor(transaction_lock, allow_unrecorded_retirement=True)
        anchor["journal"] = journal_anchor
        _write_lock_anchor(transaction_lock, anchor)
        phase = "quarantined"
    if phase == "quarantined":
        if _path_entry_exists(journal):
            raise RuntimeError(f"foreign journal blocks retirement: {journal}")
        outcome = _retire_quarantined_entry(
            quarantine,
            expected,
            kind="file",
            transaction_lock=transaction_lock,
        )
        retirement = {**retirement, "phase": outcome}
        journal_anchor["retirement"] = retirement
        anchor = _read_lock_anchor(transaction_lock)
        anchor["journal"] = journal_anchor
        _write_lock_anchor(transaction_lock, anchor)
        phase = outcome
    if phase == "gone":
        if not (
            os.name == "nt" or _uses_private_trash_retirement()
        ) or _path_entry_exists(quarantine):
            raise RuntimeError(f"shard journal retirement changed: {quarantine}")
    elif phase == "debt_recorded":
        if not _recorded_retired_entry(transaction_lock, quarantine, kind="file"):
            raise RuntimeError(f"shard journal retirement changed: {quarantine}")
    else:
        raise RuntimeError("shard journal retirement phase is invalid")
    _validate_manifest_commit(committed_payload, transaction_lock)
    anchor = _read_lock_anchor(transaction_lock)
    manifest = anchor.get("manifest")
    if committed_payload.get("state") == "publishing" and (
        not isinstance(manifest, dict) or manifest.get("phase") != "committed"
    ):
        raise RuntimeError("manifest commit evidence is incomplete")
    if (
        committed_payload.get("state") != "publishing"
        and manifest is not None
        and (not isinstance(manifest, dict) or manifest.get("phase") != "committed")
    ):
        raise RuntimeError("manifest commit evidence is incomplete")
    anchor["journal"] = None
    anchor["owner"] = None
    anchor["temporaries"] = []
    anchor["manifest"] = None
    _write_lock_anchor(transaction_lock, anchor)
    return True


def _remove_shard_transaction(
    journal: Path,
    expected_identity: FileIdentity,
    transaction_lock: BinaryIO,
) -> None:
    if not _path_entry_exists(journal) or file_identity(journal) != expected_identity:
        raise RuntimeError(f"shard build transaction identity changed: {journal}")
    payload = _read_json_object(journal, "build transaction")
    record = _anchored_file_record(journal, payload)
    anchor = _read_lock_anchor(transaction_lock)
    accepted = anchor.get("journal")
    if not isinstance(accepted, dict) or not any(
        _record_matches_payload(journal, candidate, payload)
        for candidate in accepted.get("accepted", [])
    ):
        raise RuntimeError(f"shard build transaction identity changed: {journal}")
    _validate_manifest_commit(payload, transaction_lock)
    anchor = _read_lock_anchor(transaction_lock)
    anchor["journal"] = {
        "phase": "retiring",
        "accepted": [record],
        "retirement": {
            "journal_path": str(journal),
            "quarantine_path": str(_retired_entry_path(journal.parent, "file")),
            "identity": record["identity"],
            "phase": "intended",
            "payload": payload,
            "payload_fingerprint": stable_fingerprint(payload),
        },
    }
    _write_lock_anchor(transaction_lock, anchor)
    _complete_journal_retirement(journal, transaction_lock)


def _recover_anchored_transaction_temporaries(
    journal: Path, transaction_lock: BinaryIO
) -> None:
    anchor = _read_lock_anchor(transaction_lock)
    temporaries = anchor.get("temporaries", [])
    if not isinstance(temporaries, list) or len(temporaries) > 1:
        raise RuntimeError("transaction temporary anchor is invalid")
    if not temporaries:
        return
    record = temporaries[0]
    if not isinstance(record, dict):
        raise RuntimeError("transaction temporary anchor is invalid")
    temporary = Path(str(record.get("path", "")))
    if (
        temporary.parent != journal.parent
        or re.fullmatch(
            rf"\.{re.escape(journal.name)}\.[0-9a-f]{{32}}\.partial",
            temporary.name,
        )
        is None
    ):
        raise RuntimeError("transaction temporary anchor path is invalid")
    phase = record.get("phase")
    if phase == "initializing":
        if _path_entry_exists(temporary):
            raise RuntimeError(
                f"shard transaction temporary identity changed: {temporary}"
            )
        anchor = _read_lock_anchor(transaction_lock)
        anchor["temporaries"] = []
        _write_lock_anchor(transaction_lock, anchor)
        return
    elif phase == "created":
        expected = record.get("instance")
        if not _path_entry_exists(temporary):
            raise RuntimeError(f"shard transaction temporary disappeared: {temporary}")
        if (
            _path_is_link_like(temporary)
            or not temporary.is_file()
            or _entry_instance(temporary) != expected
        ):
            raise RuntimeError(
                f"shard transaction temporary identity changed: {temporary}"
            )
        if file_identity(temporary).size:
            try:
                payload = _read_json_object(temporary, "transaction temporary")
            except RuntimeError:
                raise RuntimeError(
                    f"shard transaction temporary identity changed: {temporary}"
                ) from None
            if record.get("fingerprint") != stable_fingerprint(payload):
                raise RuntimeError(
                    f"shard transaction temporary identity changed: {temporary}"
                )
        if _entry_instance(temporary) != expected:
            raise RuntimeError(
                f"shard transaction temporary identity changed: {temporary}"
            )
        record = {
            **record,
            "phase": "retiring",
            "identity": _file_identity_payload(temporary),
            "quarantine_path": str(_retired_entry_path(journal.parent, "file")),
            "retirement_phase": "intended",
        }
    elif phase == "active":
        if not _path_entry_exists(temporary):
            journal_anchor = anchor.get("journal")
            accepted = (
                journal_anchor.get("accepted", [])
                if isinstance(journal_anchor, dict)
                else []
            )
            if _path_entry_exists(journal):
                payload = _read_json_object(journal, "transaction")
                if any(
                    _record_matches_payload(journal, candidate, payload)
                    for candidate in accepted
                ):
                    anchor["temporaries"] = []
                    _write_lock_anchor(transaction_lock, anchor)
                    return
            raise RuntimeError(f"shard transaction temporary disappeared: {temporary}")
        if _path_entry_exists(temporary):
            try:
                payload = _read_json_object(temporary, "transaction temporary")
            except RuntimeError:
                raise RuntimeError(
                    f"shard transaction temporary identity changed: {temporary}"
                ) from None
            if not _record_matches_payload(temporary, record, payload):
                raise RuntimeError(
                    f"shard transaction temporary identity changed: {temporary}"
                )
            if not _identity_payload_matches(temporary, record.get("identity")):
                raise RuntimeError(
                    f"shard transaction temporary identity changed: {temporary}"
                )
            record = {
                **record,
                "phase": "retiring",
                "quarantine_path": str(_retired_entry_path(journal.parent, "file")),
                "retirement_phase": "intended",
            }
    elif phase != "retiring":
        raise RuntimeError("transaction temporary anchor phase is invalid")

    if record.get("phase") == "retiring":
        quarantine = Path(str(record.get("quarantine_path", "")))
        expected = _identity_from_payload(record.get("identity"))
        retirement_phase = record.get("retirement_phase")
        if not _retirement_path_matches(
            journal.parent, quarantine, kind="file"
        ) or retirement_phase not in {
            "intended",
            "quarantined",
            "gone",
            "debt_recorded",
        }:
            raise RuntimeError("transaction temporary retirement anchor is invalid")
        anchor = _read_lock_anchor(transaction_lock, allow_unrecorded_retirement=True)
        anchor["temporaries"] = [record]
        _write_lock_anchor(transaction_lock, anchor)
        if retirement_phase == "intended":
            if _path_entry_exists(temporary):
                _move_entry_to_quarantine_exact(
                    temporary, quarantine, expected, kind="file"
                )
            elif not _entry_matches_expected(quarantine, expected, kind="file"):
                if not _recorded_retired_entry(
                    transaction_lock, quarantine, kind="file"
                ):
                    raise RuntimeError(
                        f"shard transaction temporary disappeared: {temporary}"
                    )
            record = {**record, "retirement_phase": "quarantined"}
            anchor = _read_lock_anchor(
                transaction_lock, allow_unrecorded_retirement=True
            )
            anchor["temporaries"] = [record]
            _write_lock_anchor(transaction_lock, anchor)
            retirement_phase = "quarantined"
        if retirement_phase == "quarantined":
            outcome = _retire_quarantined_entry(
                quarantine,
                expected,
                kind="file",
                transaction_lock=transaction_lock,
            )
            record = {**record, "retirement_phase": outcome}
            anchor = _read_lock_anchor(transaction_lock)
            anchor["temporaries"] = [record]
            _write_lock_anchor(transaction_lock, anchor)
            retirement_phase = outcome
        if retirement_phase == "gone":
            if not (
                os.name == "nt" or _uses_private_trash_retirement()
            ) or _path_entry_exists(quarantine):
                raise RuntimeError(
                    f"shard transaction temporary identity changed: {quarantine}"
                )
        elif retirement_phase == "debt_recorded":
            if not _recorded_retired_entry(transaction_lock, quarantine, kind="file"):
                raise RuntimeError(
                    f"shard transaction temporary identity changed: {quarantine}"
                )
        else:
            raise RuntimeError("transaction temporary retirement phase is invalid")
    else:
        raise RuntimeError("transaction temporary anchor phase is invalid")
    anchor = _read_lock_anchor(transaction_lock)
    anchor["temporaries"] = []
    _write_lock_anchor(transaction_lock, anchor)


def _after_manifest_recovery_entry_removal(_path: Path) -> None:
    """Test seam after a manifest recovery unlink and before progress advances."""


def _recover_anchored_manifest_temporary(
    manifest: Path, transaction_lock: BinaryIO
) -> None:
    anchor = _read_lock_anchor(transaction_lock)
    raw = anchor.get("manifest")
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise RuntimeError("manifest temporary anchor is invalid")
    partial = Path(str(raw.get("path", "")))
    if (
        raw.get("destination") != str(manifest)
        or partial.parent != manifest.parent
        or re.fullmatch(
            rf"\.{re.escape(manifest.name)}\.[0-9a-f]{{32}}\.partial",
            partial.name,
        )
        is None
        or not isinstance(raw.get("fingerprint"), str)
    ):
        raise RuntimeError("manifest temporary anchor path is invalid")
    phase = raw.get("phase")
    partial_exists = _path_entry_exists(partial)
    final_exists = _path_entry_exists(manifest)

    def require_payload(path: Path) -> None:
        try:
            payload = _read_json_object(path, "manifest temporary")
        except RuntimeError:
            raise RuntimeError(f"manifest temporary identity changed: {path}") from None
        if stable_fingerprint(payload) != raw.get("fingerprint"):
            raise RuntimeError(f"manifest temporary identity changed: {path}")

    retirement: dict[str, Any] | None = None
    if phase == "initializing":
        if partial_exists or final_exists:
            raise RuntimeError(f"manifest temporary identity changed: {partial}")
    elif phase == "created":
        if final_exists or not partial_exists:
            raise RuntimeError(f"manifest temporary identity changed: {manifest}")
        if (
            _path_is_link_like(partial)
            or not partial.is_file()
            or _entry_instance(partial) != raw.get("instance")
        ):
            raise RuntimeError(f"manifest temporary identity changed: {partial}")
        if file_identity(partial).size:
            require_payload(partial)
        if _entry_instance(partial) != raw.get("instance"):
            raise RuntimeError(f"manifest temporary identity changed: {partial}")
        retirement = {
            **raw,
            "phase": "retiring",
            "identity": _identity_payload(file_identity(partial)),
            "remaining": ["partial"],
            "removal_intent": None,
            "retirement_phase": None,
            "quarantine_paths": {
                "partial": str(_retired_entry_path(manifest.parent, "file"))
            },
        }
    elif phase == "active":
        expected = _identity_from_payload(raw.get("identity"))
        if partial_exists:
            if (
                _path_is_link_like(partial)
                or not partial.is_file()
                or file_identity(partial) != expected
            ):
                raise RuntimeError(f"manifest temporary identity changed: {partial}")
            require_payload(partial)
        if final_exists:
            if (
                _path_is_link_like(manifest)
                or not manifest.is_file()
                or file_identity(manifest) != expected
            ):
                raise RuntimeError(f"manifest temporary identity changed: {manifest}")
            require_payload(manifest)
        if not partial_exists and not final_exists:
            raise RuntimeError(f"manifest temporary identity changed: {partial}")
        if partial_exists:
            retirement = {
                **raw,
                "phase": "retiring",
                "remaining": (["manifest", "partial"] if final_exists else ["partial"]),
                "removal_intent": None,
                "retirement_phase": None,
                "quarantine_paths": {
                    label: str(_retired_entry_path(manifest.parent, "file"))
                    for label in (
                        ["manifest", "partial"] if final_exists else ["partial"]
                    )
                },
            }
        # A missing private name with the exact final identity proves the
        # no-replace publication completed.  Persist that fact before returning;
        # only journal retirement may clear committed manifest evidence.
        else:
            published = {
                **raw,
                "phase": "published",
                "identity": _identity_payload(expected),
            }
            anchor = _read_lock_anchor(transaction_lock)
            anchor["manifest"] = published
            _write_lock_anchor(transaction_lock, anchor)
            return
    elif phase in {"published", "committed"}:
        expected = _identity_from_payload(raw.get("identity"))
        if (
            partial_exists
            or not final_exists
            or not _identity_payload_matches(manifest, _identity_payload(expected))
        ):
            raise RuntimeError(f"manifest temporary identity changed: {manifest}")
        require_payload(manifest)
        if phase == "published":
            current = _read_lock_anchor(transaction_lock)
            if current.get("journal") is None:
                raise RuntimeError(
                    "published manifest is missing journal commit evidence"
                )
        return
    elif phase == "retiring":
        retirement = raw
    else:
        raise RuntimeError("manifest temporary anchor phase is invalid")
    if retirement is not None:
        remaining = retirement.get("remaining")
        intent = retirement.get("removal_intent")
        retirement_phase = retirement.get("retirement_phase")
        quarantine_paths = retirement.get("quarantine_paths")
        if isinstance(remaining, list) and not isinstance(quarantine_paths, dict):
            quarantine_paths = {
                str(label): str(_retired_entry_path(manifest.parent, "file"))
                for label in remaining
            }
            retirement["quarantine_paths"] = quarantine_paths
            retirement.setdefault("retirement_phase", None)
            anchor = _read_lock_anchor(transaction_lock)
            anchor["manifest"] = retirement
            _write_lock_anchor(transaction_lock, anchor)
            retirement_phase = retirement.get("retirement_phase")
        if (
            not isinstance(remaining, list)
            or any(label not in {"manifest", "partial"} for label in remaining)
            or len(set(remaining)) != len(remaining)
            or (intent is not None and intent not in {"manifest", "partial"})
            or (intent is not None and (not remaining or intent != remaining[0]))
            or not isinstance(quarantine_paths, dict)
            or set(quarantine_paths) != set(remaining)
            or retirement_phase
            not in {None, "intended", "quarantined", "gone", "debt_recorded"}
        ):
            raise RuntimeError("manifest temporary retirement progress is invalid")
        expected = _identity_from_payload(retirement.get("identity"))
        anchor = _read_lock_anchor(transaction_lock)
        anchor["manifest"] = retirement
        _write_lock_anchor(transaction_lock, anchor)
        paths = {"manifest": manifest, "partial": partial}
        while remaining:
            label = str(remaining[0])
            path = paths[label]
            if retirement.get("removal_intent") is None:
                retirement["removal_intent"] = label
                retirement["retirement_phase"] = "intended"
                anchor = _read_lock_anchor(transaction_lock)
                anchor["manifest"] = retirement
                _write_lock_anchor(transaction_lock, anchor)
            quarantine = Path(str(cast(dict[str, object], quarantine_paths)[label]))
            if not _retirement_path_matches(manifest.parent, quarantine, kind="file"):
                raise RuntimeError("manifest temporary retirement progress is invalid")
            if retirement.get("retirement_phase") == "intended" and _path_entry_exists(
                path
            ):
                if (
                    _path_is_link_like(path)
                    or not path.is_file()
                    or file_identity(path) != expected
                ):
                    raise RuntimeError(f"manifest temporary identity changed: {path}")
                if expected.size:
                    require_payload(path)
                if file_identity(path) != expected:
                    raise RuntimeError(f"manifest temporary identity changed: {path}")
                _move_entry_to_quarantine_exact(path, quarantine, expected, kind="file")
            elif retirement.get("retirement_phase") == "intended" and not (
                _entry_matches_expected(quarantine, expected, kind="file")
                or _recorded_retired_entry(transaction_lock, quarantine, kind="file")
            ):
                raise RuntimeError(f"manifest temporary identity changed: {path}")
            if retirement.get("retirement_phase") == "intended":
                retirement["retirement_phase"] = "quarantined"
                anchor = _read_lock_anchor(
                    transaction_lock, allow_unrecorded_retirement=True
                )
                anchor["manifest"] = retirement
                _write_lock_anchor(transaction_lock, anchor)
            if retirement.get("retirement_phase") == "quarantined":
                outcome = _retire_quarantined_entry(
                    quarantine,
                    expected,
                    kind="file",
                    transaction_lock=transaction_lock,
                )
                retirement["retirement_phase"] = outcome
                anchor = _read_lock_anchor(transaction_lock)
                anchor["manifest"] = retirement
                _write_lock_anchor(transaction_lock, anchor)
            if retirement.get("retirement_phase") == "gone":
                if not (
                    os.name == "nt" or _uses_private_trash_retirement()
                ) or _path_entry_exists(quarantine):
                    raise RuntimeError(
                        f"manifest temporary identity changed: {quarantine}"
                    )
            elif retirement.get("retirement_phase") == "debt_recorded":
                if not _recorded_retired_entry(
                    transaction_lock, quarantine, kind="file"
                ):
                    raise RuntimeError(
                        f"manifest temporary identity changed: {quarantine}"
                    )
            else:
                raise RuntimeError("manifest temporary retirement progress is invalid")
            _after_manifest_recovery_entry_removal(path)
            remaining.pop(0)
            cast(dict[str, object], quarantine_paths).pop(label)
            retirement["removal_intent"] = None
            retirement["retirement_phase"] = None
            anchor = _read_lock_anchor(transaction_lock)
            anchor["manifest"] = retirement
            _write_lock_anchor(transaction_lock, anchor)
    anchor = _read_lock_anchor(transaction_lock)
    anchor["manifest"] = None
    _write_lock_anchor(transaction_lock, anchor)


def _normalize_transaction_anchor(
    journal: Path,
    payload: Mapping[str, Any],
    identity: FileIdentity,
    transaction_lock: BinaryIO,
) -> None:
    _recover_anchored_transaction_temporaries(journal, transaction_lock)
    record = _anchored_file_record(journal, payload)
    if _identity_payload(identity) != record["identity"]:
        raise RuntimeError(f"shard build transaction identity changed: {journal}")
    anchor = _read_lock_anchor(transaction_lock)
    anchor["journal"] = {"phase": "active", "accepted": [record]}
    anchor["temporaries"] = []
    if payload.get("state") in {"building", "publishing"}:
        anchor["owner"] = None
    _write_lock_anchor(transaction_lock, anchor)


def _recover_empty_transaction_anchor(
    journal: Path, transaction_lock: BinaryIO
) -> None:
    if _complete_journal_retirement(journal, transaction_lock):
        return
    _recover_anchored_transaction_temporaries(journal, transaction_lock)
    anchor = _read_lock_anchor(transaction_lock)
    journal_anchor = anchor.get("journal")
    temporaries = anchor.get("temporaries", [])
    if journal_anchor is None and not temporaries:
        return
    if isinstance(journal_anchor, dict) and journal_anchor.get("phase") in {
        "retiring",
        "transition",
    }:
        anchor["journal"] = None
        anchor["owner"] = None
        _write_lock_anchor(transaction_lock, anchor)
        return
    raise RuntimeError("incomplete shard transaction anchor is ambiguous")


def _after_public_artifact_rollback_removal(_path: Path) -> None:
    """Test seam after public shard rollback unlink and before progress advances."""


def _mark_publication_rolled_back(
    transaction: dict[str, Any],
    journal: Path,
    journal_identity: FileIdentity,
    transaction_lock: BinaryIO,
) -> FileIdentity:
    if (
        transaction.get("state") != "publishing"
        or transaction.get("rollback_intent") is not None
        or transaction.get("directory_rollback_intent") is not None
        or transaction.get("work_rollback_intent") is not None
        or _path_entry_exists(Path(str(transaction.get("work", ""))))
    ):
        raise RuntimeError("shard publication rollback is incomplete")
    output = Path(str(transaction.get("output_root", "")))
    artifacts = transaction.get("artifacts")
    directories = transaction.get("published_directories")
    if not isinstance(artifacts, list) or not isinstance(directories, list):
        raise RuntimeError("shard publication rollback is incomplete")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError("shard publication rollback is incomplete")
        final = output.joinpath(*_canonical_entry_parts(artifact.get("relative")))
        if _path_entry_exists(final):
            raise RuntimeError(f"published shard remains after rollback: {final}")
    for record in directories:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError("shard publication rollback is incomplete")
        directory = Path(record["path"])
        if _path_entry_exists(directory):
            raise RuntimeError(
                f"published directory remains after rollback: {directory}"
            )
    transaction["state"] = "rolled_back"
    return _replace_shard_transaction(
        journal, transaction, journal_identity, transaction_lock
    )


def _public_file_retirement_path(
    output: Path, transaction_id: object, relative: str
) -> Path:
    token = hashlib.sha256(f"{transaction_id}\0{relative}".encode("utf-8")).hexdigest()[
        :32
    ]
    if _uses_private_trash_retirement():
        return _private_trash_path(output) / f"{token}.file"
    return output / f".bitguard-retired-{token}.file"


def _validate_publication_and_rollback(
    transaction: dict[str, Any],
    journal: Path,
    journal_identity: FileIdentity,
    transaction_lock: BinaryIO,
) -> FileIdentity:
    output = Path(str(transaction["output_root"]))
    work = Path(str(transaction["work"]))
    artifacts = transaction.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("shard publication artifact contract is invalid")
    rollback_removed = cast(list[str], transaction.get("rollback_removed"))
    rollback_intent = transaction.get("rollback_intent")
    directory_rollback_removed = cast(
        list[str], transaction.get("directory_rollback_removed")
    )
    directory_rollback_intent = transaction.get("directory_rollback_intent")
    if (
        not isinstance(rollback_removed, list)
        or len(set(rollback_removed)) != len(rollback_removed)
        or (rollback_intent is not None and not isinstance(rollback_intent, str))
        or not isinstance(directory_rollback_removed, list)
        or len(set(directory_rollback_removed)) != len(directory_rollback_removed)
        or any(not isinstance(value, str) for value in directory_rollback_removed)
        or (
            directory_rollback_intent is not None
            and not isinstance(directory_rollback_intent, dict)
        )
    ):
        raise RuntimeError("shard publication rollback progress is invalid")
    if directory_rollback_intent is not None:
        intent_identity = directory_rollback_intent.get("identity")
        if (
            not isinstance(directory_rollback_intent.get("path"), str)
            or not isinstance(intent_identity, list)
            or len(intent_identity) != 3
            or any(type(value) is not int for value in intent_identity)
            or not isinstance(directory_rollback_intent.get("quarantine_path"), str)
            or directory_rollback_intent.get("phase") not in {"intended", "quarantined"}
        ):
            raise RuntimeError("shard directory rollback progress is invalid")
    observed: list[tuple[str, Path, Path, FileIdentity, bool]] = []
    final_presence: list[bool] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError("shard publication artifact contract is invalid")
        parts = _canonical_entry_parts(artifact.get("relative"))
        relative = PurePosixPath(*parts).as_posix()
        staged = work.joinpath("staged", *parts)
        final = output.joinpath(*parts)
        identity = _identity_from_payload(artifact.get("identity"))
        final_exists = _path_entry_exists(final)
        staged_exists = _path_entry_exists(staged)
        if final_exists and (
            _path_is_link_like(final)
            or not final.is_file()
            or file_identity(final) != identity
            or final.stat().st_size != artifact.get("size")
            or _sha256_file(final) != artifact.get("sha256")
        ):
            raise RuntimeError(f"published shard identity changed: {final}")
        if staged_exists and (
            _path_is_link_like(staged)
            or not staged.is_file()
            or file_identity(staged) != identity
        ):
            raise RuntimeError(f"staged shard identity changed: {staged}")
        if (
            final_exists
            and staged_exists
            and _entry_instance(final) != _entry_instance(staged)
        ):
            raise RuntimeError(f"published shard hard-link identity changed: {final}")
        if relative in rollback_removed and final_exists:
            raise RuntimeError(f"published shard reappeared after rollback: {final}")
        if (
            not final_exists
            and not staged_exists
            and relative not in rollback_removed
            and relative != rollback_intent
        ):
            raise RuntimeError("published shard prefix is missing an owned artifact")
        observed.append((relative, staged, final, identity, final_exists))
        final_presence.append(final_exists)
    actual_prefix = 0
    while actual_prefix < len(final_presence) and final_presence[actual_prefix]:
        actual_prefix += 1
    if any(final_presence[actual_prefix:]):
        raise RuntimeError("published shard prefix is non-contiguous")
    durable_prefix = int(transaction["published_prefix"])
    if (
        durable_prefix > actual_prefix
        and not rollback_removed
        and rollback_intent is None
    ):
        raise RuntimeError("published shard prefix is missing a durable artifact")
    directories = transaction.get("published_directories")
    if not isinstance(directories, list):
        raise RuntimeError("published directory contract is invalid")
    directory_records: list[tuple[Path, list[int]]] = []
    for record in directories:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError("published directory contract is invalid")
        directory = Path(record["path"])
        try:
            directory.relative_to(output)
        except ValueError as exc:
            raise RuntimeError("published directory escapes output root") from exc
        identity_value = record.get("identity")
        if not isinstance(identity_value, list) or len(identity_value) != 3:
            raise RuntimeError(f"published directory identity unavailable: {directory}")
        if _path_entry_exists(directory) and (
            _path_is_link_like(directory)
            or not directory.is_dir()
            or _entry_instance(directory) != identity_value
        ):
            raise RuntimeError(f"published directory identity changed: {directory}")
        directory_records.append((directory, identity_value))
    directory_intent = transaction.get("directory_intent")
    if directory_intent is not None:
        if not isinstance(directory_intent, dict):
            raise RuntimeError("published directory intent is invalid")
        directory = Path(str(directory_intent.get("path", "")))
        try:
            directory_relative = directory.relative_to(output)
        except ValueError as exc:
            raise RuntimeError(
                "published directory intent escapes output root"
            ) from exc
        if not directory_relative.parts or directory.parent == directory:
            raise RuntimeError("published directory intent path is invalid")
        parent_identity = directory_intent.get("parent_identity")
        parent_missing_with_progress = (
            not _path_entry_exists(directory.parent)
            and isinstance(directory_rollback_intent, dict)
            and directory_rollback_intent.get("path") == str(directory.parent)
            and directory_rollback_intent.get("identity") == parent_identity
        ) or str(directory.parent) in directory_rollback_removed
        if (
            not isinstance(parent_identity, list)
            or len(parent_identity) != 3
            or (
                _entry_instance(directory.parent) != parent_identity
                and not parent_missing_with_progress
            )
        ):
            raise RuntimeError(
                f"published directory parent identity changed: {directory.parent}"
            )
        staging = Path(str(directory_intent.get("staging_path", "")))
        if (
            staging.parent != directory.parent
            or re.fullmatch(
                rf"\.{re.escape(directory.name)}\.[0-9a-f]{{32}}\.directory\.partial",
                staging.name,
            )
            is None
        ):
            raise RuntimeError("published directory staging path is invalid")
        phase = directory_intent.get("phase")
        if phase not in {"initializing", "staged"}:
            raise RuntimeError("published directory intent phase is invalid")
        directory_candidates = [
            path for path in (staging, directory) if _path_entry_exists(path)
        ]
        if phase == "initializing":
            if directory_candidates:
                raise RuntimeError(
                    "published directory identity changed: "
                    f"{directory_candidates[0]}"
                )
        else:
            identity_value = directory_intent.get("identity")
            if not isinstance(identity_value, list) or len(identity_value) != 3:
                raise RuntimeError("published directory identity is invalid")
            if len(directory_candidates) == 1:
                candidate = directory_candidates[0]
                if (
                    _path_is_link_like(candidate)
                    or not candidate.is_dir()
                    or any(candidate.iterdir())
                ):
                    raise RuntimeError(
                        f"foreign published entry blocks directory recovery: {candidate}"
                    )
                if _entry_instance(candidate) != identity_value:
                    raise RuntimeError(
                        f"published directory identity changed: {candidate}"
                    )
            elif len(directory_candidates) == 0:
                progressed_candidates = [
                    candidate
                    for candidate in (staging, directory)
                    if str(candidate) in directory_rollback_removed
                    or (
                        isinstance(directory_rollback_intent, dict)
                        and directory_rollback_intent.get("path") == str(candidate)
                        and directory_rollback_intent.get("identity") == identity_value
                    )
                ]
                if len(progressed_candidates) != 1:
                    raise RuntimeError(
                        "published directory is missing without matching rollback progress"
                    )
                candidate = progressed_candidates[0]
            else:
                raise RuntimeError(f"published directory identity changed: {directory}")
            directory_records.append((candidate, cast(list[int], identity_value)))

    ordered_directory_records: list[tuple[Path, list[int]]] = []
    directory_identities: dict[str, list[int]] = {}
    for directory, identity_value in reversed(directory_records):
        value = str(directory)
        prior = directory_identities.get(value)
        if prior is not None:
            if prior != identity_value:
                raise RuntimeError(f"published directory identity changed: {directory}")
            continue
        directory_identities[value] = identity_value
        ordered_directory_records.append((directory, identity_value))
    expected_directory_removals = [
        str(directory) for directory, _ in ordered_directory_records
    ]
    if (
        directory_rollback_removed
        != expected_directory_removals[: len(directory_rollback_removed)]
    ):
        raise RuntimeError("shard directory rollback progress is invalid")
    next_directory = (
        expected_directory_removals[len(directory_rollback_removed)]
        if len(directory_rollback_removed) < len(expected_directory_removals)
        else None
    )
    if directory_rollback_intent is not None and (
        directory_rollback_intent.get("path") != next_directory
        or directory_rollback_intent.get("identity")
        != directory_identities.get(str(next_directory))
    ):
        raise RuntimeError("shard directory rollback progress is invalid")
    for index, (directory, identity_value) in enumerate(ordered_directory_records):
        if _path_entry_exists(directory):
            if (
                _path_is_link_like(directory)
                or not directory.is_dir()
                or _entry_instance(directory) != identity_value
            ):
                raise RuntimeError(f"published directory identity changed: {directory}")
        elif index >= len(directory_rollback_removed) and (
            directory_rollback_intent is None
            or directory_rollback_intent.get("path") != str(directory)
        ):
            raise RuntimeError(
                "published directory is missing without matching rollback progress: "
                f"{directory}"
            )
    expected_files = {final for _, _, final, _, present in observed if present}
    dataset_root = Path(str(transaction["dataset_root"]))
    if _path_entry_exists(dataset_root):
        for candidate in dataset_root.rglob("*"):
            if _path_is_link_like(candidate):
                raise RuntimeError(
                    f"foreign published entry blocks recovery: {candidate}"
                )
            if candidate.is_file() and candidate not in expected_files:
                raise RuntimeError(
                    f"foreign published entry blocks recovery: {candidate}"
                )
    removed = False
    by_relative = {
        relative: (staged, final, identity, present)
        for relative, staged, final, identity, present in observed
    }
    if rollback_intent is not None and rollback_intent not in by_relative:
        raise RuntimeError("shard publication rollback intent is invalid")
    rollback_candidates: list[str] = []
    if rollback_intent is not None:
        rollback_candidates.append(rollback_intent)
    rollback_candidates.extend(
        relative
        for relative, _, _, _, present in reversed(observed)
        if present and relative not in rollback_removed and relative != rollback_intent
    )
    for artifact_relative in rollback_candidates:
        staged, final, identity, _ = by_relative[artifact_relative]
        if transaction.get("rollback_intent") is None:
            transaction["rollback_intent"] = artifact_relative
            journal_identity = _replace_shard_transaction(
                journal, transaction, journal_identity, transaction_lock
            )
        if _path_entry_exists(final):
            if (
                _path_is_link_like(final)
                or not final.is_file()
                or file_identity(final) != identity
            ):
                raise RuntimeError(f"published shard identity changed: {final}")
            if os.name == "nt" and not _uses_private_trash_retirement():
                removed = _unlink_file_if_identity(final, identity) or removed
            else:
                quarantine = _public_file_retirement_path(
                    output, transaction.get("transaction_id"), artifact_relative
                )
                if _path_entry_exists(quarantine):
                    raise RuntimeError(
                        f"published shard retirement path already exists: {quarantine}"
                    )
                _move_entry_to_quarantine_exact(
                    final, quarantine, identity, kind="file"
                )
                _retire_regular_file_without_pathname_delete(
                    quarantine, identity, transaction_lock
                )
                if _path_entry_exists(final):
                    raise RuntimeError(
                        f"foreign published entry blocks shard rollback: {final}"
                    )
                removed = True
            _fsync_directory(final.parent)
            _after_public_artifact_rollback_removal(final)
        elif transaction.get("rollback_intent") != artifact_relative:
            raise RuntimeError(f"published shard identity changed: {final}")
        elif _uses_private_trash_retirement():
            quarantine = _public_file_retirement_path(
                output, transaction.get("transaction_id"), artifact_relative
            )
            if not _path_entry_exists(quarantine):
                raise RuntimeError(f"published shard identity changed: {final}")
            observed_identity = file_identity(quarantine)
            if (
                _path_is_link_like(quarantine)
                or not quarantine.is_file()
                or (
                    observed_identity.device,
                    observed_identity.inode,
                    observed_identity.mode,
                )
                != (identity.device, identity.inode, identity.mode)
                or observed_identity.size not in {0, identity.size}
            ):
                raise RuntimeError(
                    f"published shard retirement identity changed: {quarantine}"
                )
            _retire_regular_file_without_pathname_delete(
                quarantine, identity, transaction_lock
            )
            removed = True
        rollback_removed.append(artifact_relative)
        transaction["rollback_intent"] = None
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
    for directory, identity_value in ordered_directory_records[
        len(directory_rollback_removed) :
    ]:
        if transaction.get("directory_rollback_intent") is None:
            quarantine = _retired_entry_path(output, "directory")
            transaction["directory_rollback_intent"] = {
                "path": str(directory),
                "identity": identity_value,
                "quarantine_path": str(quarantine),
                "phase": "intended",
            }
            journal_identity = _replace_shard_transaction(
                journal, transaction, journal_identity, transaction_lock
            )
        current_intent = transaction.get("directory_rollback_intent")
        if (
            not isinstance(current_intent, dict)
            or current_intent.get("path") != str(directory)
            or current_intent.get("identity") != identity_value
        ):
            raise RuntimeError("shard directory rollback progress is invalid")
        quarantine = Path(str(current_intent.get("quarantine_path", "")))
        if not _retirement_path_matches(output, quarantine, kind="directory"):
            raise RuntimeError("shard directory rollback quarantine is invalid")
        if current_intent.get("phase") == "intended":
            if _path_entry_exists(directory):
                if (
                    _path_is_link_like(directory)
                    or not directory.is_dir()
                    or _entry_instance(directory) != identity_value
                ):
                    raise RuntimeError(
                        f"published directory identity changed: {directory}"
                    )
                if _path_entry_exists(quarantine):
                    raise RuntimeError(
                        f"published directory quarantine already exists: {quarantine}"
                    )
                _before_public_directory_rollback_quarantine_boundary(directory)
                try:
                    _move_entry_to_quarantine_exact(
                        directory, quarantine, identity_value, kind="directory"
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"published directory identity changed: {directory}"
                    ) from exc
            elif not _path_entry_exists(quarantine):
                raise RuntimeError(
                    "published directory disappeared without quarantined progress: "
                    f"{directory}"
                )
            if (
                _path_is_link_like(quarantine)
                or not quarantine.is_dir()
                or _entry_instance(quarantine) != identity_value
            ):
                if not _path_entry_exists(directory) and _path_entry_exists(quarantine):
                    try:
                        _rename_directory_no_replace(quarantine, directory)
                        _fsync_directory(directory.parent)
                        if quarantine.parent != directory.parent:
                            _fsync_directory(quarantine.parent)
                    except BaseException:
                        pass
                raise RuntimeError(
                    f"published directory identity changed: {quarantine}"
                )
            current_intent = {**current_intent, "phase": "quarantined"}
            transaction["directory_rollback_intent"] = current_intent
            journal_identity = _replace_shard_transaction(
                journal, transaction, journal_identity, transaction_lock
            )
        elif current_intent.get("phase") != "quarantined":
            raise RuntimeError("shard directory rollback progress is invalid")
        if _path_entry_exists(quarantine):
            if (
                _path_is_link_like(quarantine)
                or not quarantine.is_dir()
                or _entry_instance(quarantine) != identity_value
            ):
                raise RuntimeError(
                    f"published directory identity changed: {quarantine}"
                )
            try:
                if _entry_instance(quarantine) != identity_value:
                    raise RuntimeError(
                        f"published directory identity changed: {quarantine}"
                    )
                if os.name == "nt" and not _uses_private_trash_retirement():
                    _rmdir_if_identity(quarantine, identity_value)
                else:
                    _retire_empty_directory_without_pathname_delete(
                        quarantine, identity_value, transaction_lock
                    )
                _fsync_directory(quarantine.parent)
                removed = True
                _after_public_directory_rollback_removal_boundary(directory)
            except OSError:
                if any(quarantine.iterdir()):
                    raise RuntimeError(
                        "foreign published entry blocks directory rollback: "
                        f"{quarantine}"
                    ) from None
                raise
        else:
            _fsync_directory(directory.parent)
        directory_rollback_removed.append(str(directory))
        transaction["directory_rollback_intent"] = None
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
    if removed:
        _fsync_directory(output)
    return journal_identity


def _recover_shard_transaction(
    journal: Path,
    *,
    output: Path,
    request_fingerprint: str,
    dataset: str,
    transaction_lock: BinaryIO,
) -> None:
    manifest_path = output / "shard_manifest.json"
    if _complete_journal_retirement(journal, transaction_lock):
        return
    _recover_anchored_manifest_temporary(manifest_path, transaction_lock)
    if _journal_replacement_record(journal, transaction_lock) is not None:
        _complete_journal_replacement(journal, transaction_lock)
    if not _path_entry_exists(journal):
        _recover_empty_transaction_anchor(journal, transaction_lock)
        return
    transaction, journal_identity = _load_shard_transaction(
        journal,
        output=output,
        request_fingerprint=request_fingerprint,
        dataset=dataset,
        transaction_lock=transaction_lock,
    )
    _normalize_transaction_anchor(
        journal, transaction, journal_identity, transaction_lock
    )
    dataset_root = Path(str(transaction["dataset_root"]))
    work = Path(str(transaction["work"]))
    if _path_entry_exists(manifest_path):
        if _path_entry_exists(work):
            raise RuntimeError(
                "ambiguous completed shard transaction still owns private work"
            )
        if transaction["state"] == "publishing":
            for artifact in cast(list[dict[str, Any]], transaction["artifacts"]):
                final = output.joinpath(*_canonical_entry_parts(artifact["relative"]))
                if not _identity_payload_matches(final, artifact.get("identity")):
                    raise RuntimeError(f"published shard identity changed: {final}")
        _remove_shard_transaction(journal, journal_identity, transaction_lock)
        return
    if transaction["state"] == "publishing":
        journal_identity = _prepare_owned_shard_work_cleanup(
            transaction, journal, journal_identity, transaction_lock
        )
        journal_identity = _validate_publication_and_rollback(
            transaction, journal, journal_identity, transaction_lock
        )
        journal_identity = _cleanup_owned_shard_work(
            transaction, journal, journal_identity, transaction_lock
        )
        journal_identity = _mark_publication_rolled_back(
            transaction, journal, journal_identity, transaction_lock
        )
        _remove_shard_transaction(journal, journal_identity, transaction_lock)
        return
    if _path_entry_exists(dataset_root) and (
        _is_link_like(dataset_root)
        or not dataset_root.is_dir()
        or any(dataset_root.rglob("*"))
    ):
        raise RuntimeError(
            "incomplete published shard output blocks automatic build recovery"
        )
    journal_identity = _prepare_owned_shard_work_cleanup(
        transaction, journal, journal_identity, transaction_lock
    )
    journal_identity = _cleanup_owned_shard_work(
        transaction, journal, journal_identity, transaction_lock
    )
    _remove_shard_transaction(journal, journal_identity, transaction_lock)


def _after_staged_write_boundary(_index: int, _partial: Path) -> None:
    """Test seam after a Parquet write while all shard bytes remain private."""


def _after_public_shard_link_boundary(
    _index: int, _source: Path, _destination: Path
) -> None:
    """Test seam after a public hard link is durable but before source removal."""


def _after_public_shard_source_removal(
    _index: int, _source: Path, _destination: Path
) -> None:
    """Test seam after staged-name removal and before prefix progress advances."""


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _shard_schema(materialized_features: Sequence[str]) -> pa.Schema:
    fields = [
        pa.field("row_uid", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("sequence_index", pa.int64(), nullable=False),
        pa.field("device_id", pa.string(), nullable=False),
        pa.field("raw_attack", pa.string(), nullable=False),
        pa.field("behavior_label", pa.string(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=True),
    ]
    fields.extend(
        pa.field(name, pa.float32(), nullable=True) for name in materialized_features
    )
    return pa.schema(fields)


def _working_schema(materialized_features: Sequence[str]) -> pa.Schema:
    fields = list(_shard_schema(materialized_features))
    fields.insert(7, pa.field("split", pa.string(), nullable=False))
    return pa.schema(fields)


def _validate_features(
    selected_features: Sequence[str], *, subject: str = "selected_features"
) -> tuple[str, ...]:
    features = tuple(str(name) for name in selected_features)
    if not features:
        raise ValueError(f"{subject} must not be empty")
    if any(not name for name in features):
        raise ValueError(f"{subject} must contain non-empty names")
    if len(set(features)) != len(features):
        raise ValueError(f"{subject} must not contain duplicates")
    collision = sorted(set(features) & _RESERVED_COLUMNS)
    if collision:
        raise ValueError(f"{subject} collide with metadata: {collision}")
    return features


def _validate_optional_features(
    values: Sequence[str], *, subject: str
) -> tuple[str, ...]:
    features = tuple(str(name) for name in values)
    if any(not name for name in features):
        raise ValueError(f"{subject} must contain non-empty names")
    if len(set(features)) != len(features):
        raise ValueError(f"{subject} must not contain duplicates")
    collision = sorted(set(features) & _RESERVED_COLUMNS)
    if collision:
        raise ValueError(f"{subject} collide with metadata: {collision}")
    return features


def _validate_materialization_contract(
    selected_features: Sequence[str],
    materialized_features: Sequence[str] | None,
    boolean_fast_path_features: Sequence[str],
    missing_boolean_fast_path_features: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, list[str]]]:
    selected = _validate_features(selected_features)
    configured = _validate_optional_features(
        boolean_fast_path_features,
        subject="boolean_fast_path_features",
    )
    missing = _validate_optional_features(
        missing_boolean_fast_path_features,
        subject="missing_boolean_fast_path_features",
    )
    if not set(missing).issubset(configured):
        raise ValueError(
            "missing_boolean_fast_path_features must be configured features"
        )
    available = tuple(name for name in configured if name not in set(missing))
    expected = selected + tuple(name for name in available if name not in selected)
    materialized = (
        expected
        if materialized_features is None
        else _validate_features(materialized_features, subject="materialized_features")
    )
    if materialized != expected:
        raise ValueError(
            "materialized_features must be selected_features followed by available "
            "Boolean fast-path features in configured order"
        )
    return (
        selected,
        materialized,
        {
            "configured_features": list(configured),
            "available_features": list(available),
            "missing_features": list(missing),
        },
    )


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
    path: Path,
    batch_rows: int,
    columns: Sequence[str] | None = None,
    tracker: _ResourceTracker | None = None,
) -> Generator[dict[str, Any], None, None]:
    with path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
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


def _close_iterators(iterators: Iterable[object]) -> list[BaseException]:
    cleanup: list[BaseException] = []
    for iterator in iterators:
        close = getattr(iterator, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as error:
            cleanup.append(error)
    return cleanup


def _raise_primary_with_cleanup(
    primary: BaseException, cleanup: Sequence[BaseException]
) -> NoReturn:
    if cleanup:
        for cleanup_error in cleanup:
            attach_cleanup_context(
                primary,
                "cleanup failure: " f"{type(cleanup_error).__name__}: {cleanup_error}",
            )
        raise primary from cleanup[0]
    raise primary


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
    iterators = [
        _iter_records(path, read_batch_rows, tracker=tracker) for path in inputs
    ]
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
        cleanup.extend(_close_iterators(iterators))
        _raise_primary_with_cleanup(primary, cleanup)
    else:
        writer.close()
        cleanup = _close_iterators(iterators)
        if cleanup:
            raise cleanup[0]
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
            destination = (
                work / f"{prefix}-merge-{pass_index:04d}-{group_index:06d}.parquet"
            )
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
                # Private trust boundary: merge runs are generated under this
                # invocation's UUID-owned work tree and are never public names.
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
        _write_run(path, buffer, schema, tracker, row_group_rows=merge_read_rows)
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


def _identity_from_status(status: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=int(status.st_dev),
        inode=int(status.st_ino),
        mode=int(status.st_mode),
        size=int(status.st_size),
        mtime_ns=int(status.st_mtime_ns),
    )


def _copy_verified_snapshot(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int | None,
    artifact: str,
) -> FileIdentity:
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = file_identity(source)
    digest = hashlib.sha256()
    copied = 0
    with (
        source.open("rb") as source_handle,
        destination.open("xb") as destination_handle,
    ):
        opened = _identity_from_status(os.fstat(source_handle.fileno()))
        while block := source_handle.read(1024 * 1024):
            _write_all(destination_handle, block)
            digest.update(block)
            copied += len(block)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
        after_handle = _identity_from_status(os.fstat(source_handle.fileno()))
    after_path = file_identity(source)
    if before != opened or before != after_handle or before != after_path:
        raise RuntimeError(f"{artifact} source identity changed during snapshot")
    if (
        digest.hexdigest() != expected_sha256
        or copied != before.size
        or (expected_size is not None and copied != expected_size)
    ):
        raise RuntimeError(f"{artifact} snapshot checksum or size mismatch")
    return before


def _assert_snapshot_source_identity(
    source: Path, expected: FileIdentity, artifact: str
) -> None:
    if _is_link_like(source) or file_identity(source) != expected:
        raise RuntimeError(f"{artifact} source identity changed after snapshot")


def _snapshot_verified_membership(
    source: Path, destination: Path, expected_sha256: str
) -> FileIdentity:
    identity = _copy_verified_snapshot(
        source,
        destination,
        expected_sha256=expected_sha256,
        expected_size=None,
        artifact="split membership",
    )
    _validate_membership_file(destination)
    return identity


def _validate_split_plan(plan: SplitPlan) -> dict[str, Any]:
    _validate_membership_file(plan.membership_path)
    try:
        manifest = read_split_manifest(plan)
        semantic = split_manifest_semantic_fingerprint(manifest)
        membership = manifest["membership"]
        counts = manifest["counts"]
        held_out = manifest["held_out"]
        held_out_attacks = held_out["attacks"]
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError("invalid split manifest for shard preparation") from exc
    expected_counts = {
        "train": plan.train_count,
        "validation": plan.validation_count,
        "test": plan.test_count,
    }
    if (
        manifest.get("strategy") != plan.strategy
        or manifest.get("fingerprint") != plan.fingerprint
        or manifest.get("semantic_fingerprint") != semantic
        or membership.get("path") != plan.membership_path.name
        or {name: int(counts.get(name, -1)) for name in _PARTITIONS} != expected_counts
        or int(membership.get("rows", -1)) != sum(expected_counts.values())
        or _sha256_file(plan.membership_path) != membership.get("sha256")
        or not isinstance(held_out_attacks, list)
        or any(not isinstance(value, str) for value in held_out_attacks)
    ):
        raise RuntimeError("split plan fingerprint or membership checksum mismatch")
    return manifest


def _join_membership(
    source_path: Path,
    split_plan: SplitPlan,
    membership_path: Path,
    work: Path,
    *,
    schema: pa.Schema,
    max_rows_per_run: int,
    merge_read_rows: int,
    held_out_attacks: frozenset[str],
    tracker: _ResourceTracker,
) -> tuple[list[Path], int, str]:
    _validate_membership_file(membership_path)
    sources = _iter_records(source_path, merge_read_rows, tracker=tracker)
    members = _iter_records(membership_path, merge_read_rows, tracker=tracker)
    source: dict[str, Any] | None = None
    member: dict[str, Any] | None = None
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
        _write_run(path, buffer, schema, tracker, row_group_rows=merge_read_rows)
        joined_runs.append(path)
        buffer.clear()

    try:
        source = next(sources, None)
        member = next(members, None)
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
                    raise RuntimeError(
                        f"duplicate split membership row_uid: {member_uid}"
                    )
            else:
                member_uid = ""
            if source is None:
                raise RuntimeError(
                    f"missing source coverage for membership UID: {member_uid}"
                )
            if member is None:
                raise RuntimeError(f"extra source coverage UID: {source_uid}")
            if source_uid < member_uid:
                raise RuntimeError(f"extra source coverage UID: {source_uid}")
            if source_uid > member_uid:
                raise RuntimeError(
                    f"missing source coverage for membership UID: {member_uid}"
                )
            split = str(member["split"])
            label = str(member["behavior_label"])
            if split not in _PARTITION_SET:
                raise RuntimeError(f"invalid split membership partition: {split}")
            if not _PATH_TOKEN.fullmatch(label):
                raise RuntimeError(
                    f"unsafe behavior label for partition path: {label!r}"
                )
            source_label = str(source["behavior_label"])
            source_attack = normalize_token(source["raw_attack"])
            sanctioned_attack_relabel = (
                split_plan.strategy == "attack"
                and split == "test"
                and label == "unknown_like"
                and source_attack in held_out_attacks
            )
            if source_label != label and not sanctioned_attack_relabel:
                if (
                    split_plan.strategy == "attack"
                    and split == "test"
                    and label == "unknown_like"
                ):
                    raise RuntimeError(
                        "attack relabel raw_attack is not a declared held-out attack "
                        f"for UID: {source_uid}"
                    )
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
        result = (joined_runs, total, uid_digest.hexdigest())
    except BaseException as primary:
        _raise_primary_with_cleanup(primary, _close_iterators((sources, members)))
    cleanup = _close_iterators((sources, members))
    if cleanup:
        raise cleanup[0]
    return result


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
    # Private trust boundary: both names live in the transaction-owned staging
    # directory, and ``final`` is absent until this writer promotes ``partial``.
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
    staged_write_index = 0

    def flush_buffer() -> None:
        nonlocal buffer, staged_write_index
        if buffer:
            assert writer is not None and partial is not None
            writer.write_table(pa.Table.from_pylist(buffer, schema=shard_schema))
            _after_staged_write_boundary(staged_write_index, partial)
            staged_write_index += 1
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
                "schema_fingerprint": stable_fingerprint(
                    _schema_descriptor(shard_schema)
                ),
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
        directory = staging / f"dataset={dataset}" / f"split={split}" / f"label={label}"
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"part-{part:08d}.parquet"
        partial = directory / f".part-{part:08d}.parquet.partial"
        writer = pq.ParquetWriter(partial, shard_schema, compression="zstd")

    records = _iter_records(sorted_path, record_batch_rows)
    try:
        for record in records:
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
        cleanup = _close_iterators((records,))
        if writer is not None:
            try:
                writer.close()
            except BaseException as cleanup_failure:
                cleanup.append(cleanup_failure)
        _raise_primary_with_cleanup(primary, cleanup)
    cleanup = _close_iterators((records,))
    if cleanup:
        raise cleanup[0]
    return sorted(entries, key=lambda entry: str(entry["path"]))


def _manifest_semantics(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema_version",
        "dataset",
        "preprocessing_fingerprint",
        "split_fingerprint",
        "selected_features",
        "materialized_features",
        "boolean_fast_path",
        "counts",
        "class_counts",
        "source_coverage",
        "coverage",
        "shard_contract",
        "schema",
        "schema_fingerprint",
        "algorithm_versions",
        "entries",
        "resource_usage",
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
    materialized_features: tuple[str, ...],
    boolean_fast_path: Mapping[str, Sequence[str]],
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
    if tracker.merge_input_rows_buffered != 0:
        raise RuntimeError("merge input rows remained buffered at manifest boundary")
    merge_input_limit = merge_fan_in * merge_read_rows
    if tracker.max_merge_input_rows_buffered > merge_input_limit:
        raise RuntimeError("merge input rows exceeded configured memory bound")
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
        "materialized_features": list(materialized_features),
        "boolean_fast_path": {
            "configured_features": list(boolean_fast_path["configured_features"]),
            "available_features": list(boolean_fast_path["available_features"]),
            "missing_features": list(boolean_fast_path["missing_features"]),
        },
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
            "max_merge_input_rows_buffered": (tracker.max_merge_input_rows_buffered),
            "merge_input_rows_buffered_limit": (merge_input_limit),
            "temporary_bytes_peak": tracker.temporary_bytes_peak,
        },
    }
    payload["fingerprint"] = stable_fingerprint(_manifest_semantics(payload))
    return payload


_SHARD_CONTRACT_FIELDS = frozenset(
    {
        "shard_target_rows",
        "record_batch_rows",
        "max_rows_per_run",
        "merge_fan_in",
        "merge_read_rows",
    }
)
_RESOURCE_USAGE_FIELDS = frozenset(
    {
        "shard_target_rows",
        "record_batch_rows",
        "configured_max_rows_per_run",
        "max_run_rows",
        "run_count",
        "merge_fan_in_limit",
        "merge_read_rows",
        "max_merge_fan_in_observed",
        "max_merge_input_rows_buffered",
        "merge_input_rows_buffered_limit",
        "temporary_bytes_peak",
    }
)


def _strict_resource_int(values: Mapping[str, Any], name: str, *, minimum: int) -> int:
    value = values.get(name)
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"invalid shard resource claim: {name}")
    return value


def _validate_resource_claims(manifest: Mapping[str, Any]) -> int:
    contract = manifest.get("shard_contract")
    resources = manifest.get("resource_usage")
    coverage = manifest.get("coverage")
    entries = manifest.get("entries")
    if (
        not isinstance(contract, Mapping)
        or set(contract) != _SHARD_CONTRACT_FIELDS
        or not isinstance(resources, Mapping)
        or set(resources) != _RESOURCE_USAGE_FIELDS
        or not isinstance(coverage, Mapping)
        or not isinstance(entries, list)
    ):
        raise RuntimeError("invalid shard resource claim structure")

    shard_target_rows = _strict_resource_int(contract, "shard_target_rows", minimum=1)
    record_batch_rows = _strict_resource_int(contract, "record_batch_rows", minimum=1)
    max_rows_per_run = _strict_resource_int(contract, "max_rows_per_run", minimum=1)
    merge_fan_in = _strict_resource_int(contract, "merge_fan_in", minimum=2)
    merge_read_rows = _strict_resource_int(contract, "merge_read_rows", minimum=1)
    coverage_rows = _strict_resource_int(coverage, "rows", minimum=1)
    resource_shard_target = _strict_resource_int(
        resources, "shard_target_rows", minimum=1
    )
    resource_record_batch = _strict_resource_int(
        resources, "record_batch_rows", minimum=1
    )
    resource_max_rows = _strict_resource_int(
        resources, "configured_max_rows_per_run", minimum=1
    )
    max_run_rows = _strict_resource_int(resources, "max_run_rows", minimum=1)
    _strict_resource_int(resources, "run_count", minimum=2)
    resource_merge_fan_in = _strict_resource_int(
        resources, "merge_fan_in_limit", minimum=2
    )
    resource_merge_read = _strict_resource_int(resources, "merge_read_rows", minimum=1)
    max_merge_fan_in = _strict_resource_int(
        resources, "max_merge_fan_in_observed", minimum=0
    )
    max_merge_rows = _strict_resource_int(
        resources, "max_merge_input_rows_buffered", minimum=1
    )
    merge_limit = _strict_resource_int(
        resources, "merge_input_rows_buffered_limit", minimum=1
    )
    temporary_peak = _strict_resource_int(resources, "temporary_bytes_peak", minimum=0)
    entry_bytes = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError("invalid shard resource claim entries")
        entry_bytes += _strict_resource_int(entry, "byte_size", minimum=1)
    expected_merge_limit = merge_fan_in * merge_read_rows
    observed_merge_limit = max(2, max_merge_fan_in) * merge_read_rows
    if (
        resource_shard_target != shard_target_rows
        or resource_record_batch != record_batch_rows
        or resource_max_rows != max_rows_per_run
        or resource_merge_fan_in != merge_fan_in
        or resource_merge_read != merge_read_rows
        or max_run_rows != coverage_rows
        or max_merge_fan_in == 1
        or max_merge_fan_in > merge_fan_in
        or max_merge_rows > observed_merge_limit
        or merge_limit != expected_merge_limit
        or temporary_peak < entry_bytes
    ):
        raise RuntimeError("inconsistent shard resource claims")
    return shard_target_rows


def load_shard_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read shard manifest: {manifest_path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SHARD_MANIFEST_SCHEMA
    ):
        raise RuntimeError(f"unsupported shard manifest schema: {manifest_path}")
    expected_algorithms = {
        "shards": SHARD_ALGORITHM,
        "coverage": COVERAGE_ALGORITHM,
    }
    if payload.get("algorithm_versions") != expected_algorithms:
        raise RuntimeError("unsupported shard manifest algorithm versions")
    try:
        fingerprint = stable_fingerprint(_manifest_semantics(payload))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid shard manifest semantics") from exc
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError("shard manifest fingerprint mismatch")
    return payload


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _canonical_entry_parts(relative: object) -> tuple[str, ...]:
    if not isinstance(relative, str):
        raise RuntimeError("non-canonical shard manifest path")
    value = relative
    if (
        not value
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RuntimeError(f"non-canonical shard manifest path: {value!r}")
    pure = PurePosixPath(value)
    native = Path(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or native.is_absolute()
        or native.drive
        or tuple(native.parts) != pure.parts
    ):
        raise RuntimeError(f"non-canonical shard manifest path: {value!r}")
    return pure.parts


def _safe_entry_path(root: Path, relative: object) -> Path:
    parts = _canonical_entry_parts(relative)
    value = str(relative)
    if _is_link_like(root) or not root.is_dir():
        raise RuntimeError(f"unsafe shard output root: {root}")
    resolved_root = root.resolve(strict=True)
    current = root
    for part in parts[:-1]:
        current = current / part
        if _is_link_like(current) or not current.is_dir():
            raise RuntimeError(f"unsafe linked shard parent directory: {current}")
    path = current / parts[-1]
    if _is_link_like(path) or not path.is_file():
        raise RuntimeError(f"shard is not a regular file: {value}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"resolved shard path escapes output root: {value}") from exc
    return resolved


def _after_public_directory_intent_boundary(_directory: Path) -> None:
    """Test seam after public-directory creation intent is durable."""


def _after_public_directory_created_anchor_boundary(_directory: Path) -> None:
    """Test seam after a new public-directory inode is durably anchored."""


def _after_public_directory_publish_boundary(_directory: Path) -> None:
    """Test seam after an anchored directory is atomically published."""


def _after_public_directory_rollback_removal_boundary(_directory: Path) -> None:
    """Test seam after directory rmdir and immediate-parent fsync."""


def _before_public_directory_rollback_quarantine_boundary(_directory: Path) -> None:
    """Test seam before atomically quarantining a rollback directory."""


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a same-volume directory without replacing a peer."""

    try:
        if os.name == "nt":
            os.rename(source, destination)
        elif sys.platform.startswith("linux"):
            renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            if (
                renameat2(
                    -100,
                    os.fsencode(source),
                    -100,
                    os.fsencode(destination),
                    1,
                )
                != 0
            ):
                error_number = ctypes.get_errno()
                raise OSError(error_number, os.strerror(error_number), destination)
        elif sys.platform == "darwin":
            renamex_np = ctypes.CDLL(None, use_errno=True).renamex_np
            renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            renamex_np.restype = ctypes.c_int
            if renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004):
                error_number = ctypes.get_errno()
                raise OSError(error_number, os.strerror(error_number), destination)
        else:
            raise RuntimeError(
                "atomic no-replace directory publication is unsupported "
                f"on {sys.platform}"
            )
    except OSError as exc:
        if isinstance(exc, FileExistsError) or exc.errno in {
            errno.EEXIST,
            errno.ENOTEMPTY,
        }:
            raise RuntimeError(
                f"immutable directory already exists: {destination}"
            ) from exc
        raise


def _ensure_durable_public_directories(
    root: Path,
    directory: Path,
    transaction: dict[str, Any],
    journal: Path,
    journal_identity: FileIdentity,
    transaction_lock: BinaryIO,
) -> tuple[list[tuple[Path, FileIdentity]], FileIdentity]:
    if _is_link_like(root) or not root.is_dir():
        raise RuntimeError(f"unsafe shard output root: {root}")
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"shard directory escapes output root: {directory}") from exc
    current = root
    created: list[tuple[Path, FileIdentity]] = []
    for part in relative.parts:
        current = current / part
        if _path_entry_exists(current):
            if _is_link_like(current) or not current.is_dir():
                raise RuntimeError(f"unsafe shard destination directory: {current}")
            continue
        parent = current.parent
        parent_identity = _entry_instance(parent)
        if parent_identity is None:
            raise RuntimeError(f"shard directory parent disappeared: {parent}")
        staging = parent / (f".{current.name}.{uuid.uuid4().hex}.directory.partial")
        transaction["directory_intent"] = {
            "phase": "initializing",
            "path": str(current),
            "staging_path": str(staging),
            "parent_identity": parent_identity,
        }
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
        _after_public_directory_intent_boundary(current)
        if _path_entry_exists(current) or _path_entry_exists(staging):
            raise RuntimeError(f"unsafe shard destination directory: {current}")
        staging.mkdir()
        _fsync_directory(parent)
        identity = file_identity(staging)
        transaction["directory_intent"] = {
            "phase": "staged",
            "path": str(current),
            "staging_path": str(staging),
            "parent_identity": parent_identity,
            "identity": _identity_payload(identity)[:3],
        }
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
        _after_public_directory_created_anchor_boundary(staging)
        _rename_directory_no_replace(staging, current)
        _fsync_directory(parent)
        _after_public_directory_publish_boundary(current)
        if (
            _path_entry_exists(staging)
            or not _path_entry_exists(current)
            or _entry_instance(current) != _identity_payload(identity)[:3]
        ):
            raise RuntimeError(f"published directory identity changed: {current}")
        cast(list[dict[str, Any]], transaction["published_directories"]).append(
            {
                "path": str(current),
                "identity": _identity_payload(identity)[:3],
            }
        )
        transaction["directory_intent"] = None
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
        created.append((current, identity))
    return created, journal_identity


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
    snapshot_path: Path,
    schema: pa.Schema,
    work: Path,
    run_paths: list[Path],
    max_rows_per_run: int,
    merge_read_rows: int,
    tracker: _ResourceTracker,
) -> tuple[Counter[str], dict[str, Counter[str]], Counter[str], FileIdentity]:
    schema_fingerprint = stable_fingerprint(_schema_descriptor(schema))
    if entry.get("schema_fingerprint") != schema_fingerprint:
        raise RuntimeError(
            f"shard entry schema fingerprint mismatch: {entry.get('path')}"
        )
    expected_sha256 = str(entry.get("sha256"))
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError(f"shard checksum mismatch: {entry.get('path')}")
    expected_size = int(entry.get("byte_size", -1))
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"shard byte size mismatch: {entry.get('path')}")
    source_identity = _copy_verified_snapshot(
        path,
        snapshot_path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        artifact=f"shard {entry.get('path')}",
    )
    with snapshot_path.open("rb") as handle:
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

    records = _iter_records(snapshot_path, merge_read_rows, tracker=tracker)
    try:
        for row in records:
            split = expected_split
            label = str(row["behavior_label"])
            if label != expected_label:
                raise RuntimeError(
                    f"shard partition metadata mismatch: {entry.get('path')}"
                )
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
    except BaseException as primary:
        _raise_primary_with_cleanup(primary, _close_iterators((records,)))
    cleanup = _close_iterators((records,))
    if cleanup:
        raise cleanup[0]
    flush()
    if dict(sorted(entry_sources.items())) != {
        str(name): int(value)
        for name, value in entry.get("source_coverage", {}).items()
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
    return counts, classes, sources, source_identity


def _verify_uid_coverage(
    actual_path: Path,
    *,
    manifest: Mapping[str, Any],
    membership_path: Path | None,
    batch_rows: int,
    tracker: _ResourceTracker,
) -> None:
    actual_iter = _iter_records(actual_path, batch_rows, tracker=tracker)
    expected_iter = (
        _iter_records(membership_path, batch_rows, tracker=tracker)
        if membership_path is not None
        else None
    )
    previous: str | None = None
    digest = hashlib.sha256()
    rows = 0
    try:
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
                        raise RuntimeError(
                            f"missing shard coverage UID: {expected_tuple[0]}"
                        )
                    raise RuntimeError(f"shard split or label mismatch for UID: {uid}")
        if expected_iter is not None:
            remaining = next(expected_iter, None)
            if remaining is not None:
                raise RuntimeError(
                    f"missing shard coverage UID: {remaining['row_uid']}"
                )
        coverage = manifest.get("coverage", {})
        if rows != int(coverage.get("rows", -1)):
            raise RuntimeError("shard coverage row count mismatch")
        if digest.hexdigest() != coverage.get("uid_digest"):
            raise RuntimeError("shard coverage UID digest mismatch")
    except BaseException as primary:
        _raise_primary_with_cleanup(
            primary,
            _close_iterators(
                (actual_iter,)
                if expected_iter is None
                else (actual_iter, expected_iter)
            ),
        )
    cleanup = _close_iterators(
        (actual_iter,) if expected_iter is None else (actual_iter, expected_iter)
    )
    if cleanup:
        raise cleanup[0]


def _before_verification_work_cleanup(_work: Path) -> None:
    """Test seam before verification work is identity-bound for cleanup."""


def _after_verification_work_quarantine_boundary(
    _work: Path, _quarantine: Path
) -> None:
    """Test seam after verification work enters its private quarantine."""


def _ensure_verification_trash(output: Path) -> tuple[Path, list[int]]:
    trash = output / _SHARD_VERIFICATION_TRASH
    try:
        trash.mkdir(mode=0o700)
        _fsync_directory(output)
    except FileExistsError:
        pass
    if _path_is_link_like(trash):
        raise RuntimeError(f"unsafe verification trash entry: {trash}")
    status = trash.stat(follow_symlinks=False)
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"unsafe verification trash entry: {trash}")
    if os.name != "nt":
        expected_uid = getattr(os, "geteuid", lambda: status.st_uid)()
        if (
            int(status.st_uid) != int(expected_uid)
            or stat.S_IMODE(status.st_mode) != 0o700
        ):
            raise RuntimeError(f"unsafe verification trash permissions: {trash}")
    return trash, _status_instance(status)


def _verification_cleanup_inventory(
    work: Path,
) -> list[tuple[tuple[str, ...], str, FileIdentity | list[int]]]:
    inventory: list[tuple[tuple[str, ...], str, FileIdentity | list[int]]] = []
    pending = [work]
    while pending:
        directory = pending.pop()
        for candidate in directory.iterdir():
            relative = candidate.relative_to(work).parts
            if _path_is_link_like(candidate):
                raise RuntimeError(
                    f"unsafe linked verification work entry: {candidate}"
                )
            status = candidate.stat(follow_symlinks=False)
            if stat.S_ISREG(status.st_mode):
                expected: FileIdentity | list[int] = file_identity(candidate)
                kind = "file"
            elif stat.S_ISDIR(status.st_mode):
                expected = _status_instance(status)
                kind = "directory"
                pending.append(candidate)
            else:
                raise RuntimeError(f"unsafe verification work entry type: {candidate}")
            inventory.append((relative, kind, expected))
    inventory.sort(key=lambda entry: len(entry[0]), reverse=True)
    return inventory


def _remove_windows_verification_quarantine(
    quarantine: Path,
    quarantine_identity: Sequence[int],
    inventory: Sequence[tuple[tuple[str, ...], str, FileIdentity | list[int]]],
    *,
    trash: Path,
    trash_identity: Sequence[int],
) -> None:
    if _entry_instance(trash) != list(trash_identity):
        raise RuntimeError("verification trash identity changed")
    if not _entry_matches_expected(quarantine, quarantine_identity, kind="directory"):
        raise RuntimeError("verification work quarantine identity changed")
    for relative, kind, expected in inventory:
        if not _entry_matches_expected(
            quarantine, quarantine_identity, kind="directory"
        ):
            raise RuntimeError("verification work quarantine identity changed")
        candidate = quarantine.joinpath(*relative)
        if not _entry_matches_expected(candidate, expected, kind=kind):
            raise RuntimeError(
                f"verification work quarantine entry identity changed: {candidate}"
            )
        _remove_entry_if_identity(candidate, expected, kind=kind)
    if not _entry_matches_expected(quarantine, quarantine_identity, kind="directory"):
        raise RuntimeError("verification work quarantine identity changed")
    _remove_entry_if_identity(quarantine, quarantine_identity, kind="directory")
    _fsync_directory(trash)


def _cleanup_work(
    work: Path,
    expected_identity: Sequence[int],
    primary: BaseException | None = None,
) -> None:
    if not work.exists():
        return
    try:
        _before_verification_work_cleanup(work)
        if (
            _path_is_link_like(work)
            or not work.is_dir()
            or _entry_instance(work) != list(expected_identity)
        ):
            raise RuntimeError(f"verification work identity changed: {work}")
        windows_inventory = (
            _verification_cleanup_inventory(work) if os.name == "nt" else []
        )
        if _entry_instance(work) != list(expected_identity):
            raise RuntimeError(f"verification work identity changed: {work}")
        trash, trash_identity = _ensure_verification_trash(work.parent)
        quarantine = trash / f"{uuid.uuid4().hex}.directory"
        _move_entry_to_quarantine_exact(
            work, quarantine, expected_identity, kind="directory"
        )
        _after_verification_work_quarantine_boundary(work, quarantine)
        if os.name == "nt":
            _remove_windows_verification_quarantine(
                quarantine,
                expected_identity,
                windows_inventory,
                trash=trash,
                trash_identity=trash_identity,
            )
        else:
            if not bool(getattr(shutil.rmtree, "avoids_symlink_attacks", False)):
                raise RuntimeError("safe verification cleanup is unavailable")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(trash, flags)
            try:
                if _status_instance(os.fstat(descriptor)) != trash_identity:
                    raise RuntimeError("verification trash identity changed")
                shutil.rmtree(quarantine.name, dir_fd=descriptor)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if _path_entry_exists(quarantine):
            raise RuntimeError(f"verification cleanup did not finish: {quarantine}")
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
    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not _PATH_TOKEN.fullmatch(dataset):
        raise RuntimeError("invalid shard manifest dataset token")
    _reject_stale_work(path.parent)
    if preprocessing_fingerprint is not None and manifest.get(
        "preprocessing_fingerprint"
    ) != str(preprocessing_fingerprint):
        raise RuntimeError("preprocessing fingerprint mismatch")
    split_manifest: dict[str, Any] | None = None
    if split_plan is not None:
        if manifest.get("split_fingerprint") != split_plan.fingerprint:
            raise RuntimeError("split fingerprint mismatch")
        split_manifest = _validate_split_plan(split_plan)
    try:
        boolean_contract = manifest["boolean_fast_path"]
        if not isinstance(boolean_contract, Mapping) or set(boolean_contract) != {
            "configured_features",
            "available_features",
            "missing_features",
        }:
            raise ValueError("invalid Boolean fast-path contract")
        selected_features, materialized_features, expected_boolean = (
            _validate_materialization_contract(
                manifest["selected_features"],
                manifest["materialized_features"],
                boolean_contract["configured_features"],
                boolean_contract["missing_features"],
            )
        )
        if expected_boolean != dict(boolean_contract):
            raise ValueError("Boolean fast-path availability contract mismatch")
        schema = _shard_schema(materialized_features)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid shard feature contract") from exc
    descriptor = _schema_descriptor(schema)
    if manifest.get("schema") != descriptor or manifest.get(
        "schema_fingerprint"
    ) != stable_fingerprint(descriptor):
        raise RuntimeError("shard manifest schema fingerprint mismatch")
    target_rows = _validate_resource_claims(manifest)
    work = path.parent / f".verify-shards-{uuid.uuid4().hex}.partial"
    work.mkdir(mode=0o700)
    work_identity = _entry_instance(work)
    if work_identity is None:
        raise RuntimeError(f"verification work identity is unavailable: {work}")
    tracker = _ResourceTracker(work, max_rows_per_run, merge_read_rows)
    try:
        membership_snapshot: Path | None = None
        membership_identity: FileIdentity | None = None
        if split_plan is not None and split_manifest is not None:
            membership_snapshot = work / "split-membership.snapshot.parquet"
            membership_identity = _snapshot_verified_membership(
                split_plan.membership_path,
                membership_snapshot,
                str(split_manifest["membership"]["sha256"]),
            )
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
        source_identities: list[tuple[Path, FileIdentity]] = []
        bucket_entries: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
            list
        )
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise RuntimeError("invalid shard manifest entry")
            shard = _safe_entry_path(path.parent, entry.get("path"))
            split = str(entry.get("split"))
            label = str(entry.get("label"))
            if split not in _PARTITION_SET or not _PATH_TOKEN.fullmatch(label):
                raise RuntimeError("invalid shard partition manifest entry")
            expected_relative = PurePosixPath(
                f"dataset={dataset}",
                f"split={split}",
                f"label={label}",
                PurePosixPath(str(entry.get("path"))).name,
            ).as_posix()
            if str(entry.get("path")) != expected_relative:
                raise RuntimeError(
                    f"shard path/partition mismatch: {entry.get('path')}"
                )
            bucket_entries[(split, label)].append(entry)
            resolved = shard.resolve(strict=True)
            if resolved in listed:
                raise RuntimeError(
                    f"duplicate shard manifest path: {entry.get('path')}"
                )
            listed.add(resolved)
            entry_counts, entry_classes, entry_sources, source_identity = (
                _verify_entry_and_index(
                    shard,
                    entry,
                    snapshot_path=(
                        work / "verified-shards" / f"shard-{entry_index:08d}.parquet"
                    ),
                    schema=schema,
                    work=work,
                    run_paths=run_paths,
                    max_rows_per_run=max_rows_per_run,
                    merge_read_rows=merge_read_rows,
                    tracker=tracker,
                )
            )
            source_identities.append((shard, source_identity))
            counts.update(entry_counts)
            for split, values in entry_classes.items():
                classes[split].update(values)
            sources.update(entry_sources)
        for bucket, bucket_values in bucket_entries.items():
            previous_max: tuple[Any, ...] | None = None
            for index, entry in enumerate(bucket_values):
                expected_name = f"part-{index:08d}.parquet"
                if PurePosixPath(str(entry["path"])).name != expected_name:
                    raise RuntimeError(
                        f"non-contiguous shard parts for bucket: {bucket}"
                    )
                rows = int(entry["rows"])
                if rows <= 0 or rows > target_rows:
                    raise RuntimeError(
                        f"shard target row bound exceeded: {entry['path']}"
                    )
                if index < len(bucket_values) - 1 and rows != target_rows:
                    raise RuntimeError(
                        f"non-final shard is below target rows: {entry['path']}"
                    )
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
            membership_path=membership_snapshot,
            batch_rows=merge_read_rows,
            tracker=tracker,
        )
        if (
            tracker.merge_input_rows_buffered != 0
            or tracker.max_merge_input_rows_buffered > merge_fan_in * merge_read_rows
        ):
            raise RuntimeError("verification merge input exceeded configured bound")
        dataset_root = path.parent / f"dataset={dataset}"
        if _is_link_like(dataset_root) or not dataset_root.is_dir():
            raise RuntimeError("shard dataset directory is missing or unsafe")
        for candidate in dataset_root.rglob("*"):
            if _is_link_like(candidate):
                raise RuntimeError(f"unsafe linked shard artifact: {candidate}")
        for partial in dataset_root.rglob("*.partial"):
            raise RuntimeError(f"unsafe partial shard artifact: {partial}")
        discovered = {
            candidate.resolve(strict=True)
            for candidate in dataset_root.rglob("*.parquet")
            if candidate.is_file() and not candidate.is_symlink()
        }
        if discovered != listed:
            raise RuntimeError("unlisted or missing Parquet shard artifacts")
        if split_plan is not None and membership_identity is not None:
            _assert_snapshot_source_identity(
                split_plan.membership_path,
                membership_identity,
                "split membership",
            )
        for source, identity in source_identities:
            _assert_snapshot_source_identity(source, identity, f"shard {source}")
        _cleanup_work(work, work_identity)
        return manifest
    except BaseException as primary:
        _cleanup_work(work, work_identity, primary)
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
    materialized_features: Sequence[str] | None = None,
    boolean_fast_path_features: Sequence[str] = (),
    missing_boolean_fast_path_features: Sequence[str] = (),
    shard_target_rows: int = 1_000_000,
    record_batch_rows: int = 65_536,
    max_rows_per_run: int = 65_536,
    merge_fan_in: int = 32,
    merge_read_rows: int = 1_024,
) -> ShardPlan:
    """Build immutable training/Boolean shards through bounded external merges."""

    selected, materialized, boolean_contract = _validate_materialization_contract(
        selected_features,
        materialized_features,
        boolean_fast_path_features,
        missing_boolean_fast_path_features,
    )
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
    split_manifest = _validate_split_plan(split_plan)
    held_out_attacks = frozenset(
        normalize_token(value) for value in split_manifest["held_out"]["attacks"]
    )
    output = Path(
        os.path.normcase(
            os.path.normpath(str(Path(output_dir).expanduser().resolve(strict=False)))
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    output = Path(os.path.normcase(os.path.normpath(str(output.resolve(strict=True)))))
    shard_schema = _shard_schema(materialized)
    working_schema = _working_schema(materialized)
    manifest_path = output / "shard_manifest.json"
    request_fingerprint = stable_fingerprint(
        {
            "schema": _SHARD_BUILD_TRANSACTION_SCHEMA,
            "dataset": dataset,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "split_fingerprint": split_plan.fingerprint,
            "split_manifest_fingerprint": split_manifest_semantic_fingerprint(
                split_manifest
            ),
            "selected_features": list(selected),
            "materialized_features": list(materialized),
            "boolean_fast_path": boolean_contract,
            "shard_target_rows": shard_target_rows,
            "record_batch_rows": record_batch_rows,
            "max_rows_per_run": max_rows_per_run,
            "merge_fan_in": merge_fan_in,
            "merge_read_rows": merge_read_rows,
        }
    )
    journal = _shard_transaction_path(output)
    transaction_lock = _acquire_shard_transaction_lock(
        _shard_transaction_lock_path(output)
    )
    lock_released = False
    transaction: dict[str, Any] | None = None
    journal_identity: FileIdentity | None = None
    work: Path | None = None
    tracker: _ResourceTracker | None = None
    try:
        _recover_shard_transaction(
            journal,
            output=output,
            request_fingerprint=request_fingerprint,
            dataset=dataset,
            transaction_lock=transaction_lock,
        )
        _reject_stale_work(output)
        transaction_id = uuid.uuid4().hex
        work = output / f".shards-{transaction_id}.partial"
        owner_marker = work / _SHARD_WORK_OWNER_FILE
        owner_initializing_marker = work / _SHARD_WORK_OWNER_INITIALIZING_FILE
        owner_payload = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "request_fingerprint": request_fingerprint,
        }
        transaction = {
            "schema_version": _SHARD_BUILD_TRANSACTION_SCHEMA,
            "state": "initializing",
            "transaction_id": transaction_id,
            "request_fingerprint": request_fingerprint,
            "output_root": str(output),
            "dataset_root": str(output / f"dataset={dataset}"),
            "work": str(work),
            "owner_marker": str(owner_marker),
            "owner_initializing_marker": str(owner_initializing_marker),
            "owner_payload": owner_payload,
            "work_instance": None,
            "owner_identity": None,
            "cleanup_inventory": None,
            "cleanup_removed": [],
            "cleanup_removal_intent": None,
            "cleanup_quarantine_intent": None,
            "work_rollback_intent": None,
        }
        journal_identity = _write_shard_transaction(
            journal, transaction, transaction_lock
        )
        _after_work_directory_creation_intent_boundary(work)
        work.mkdir()
        _fsync_directory(output)
        transaction["work_instance"] = _entry_instance(work)
        transaction["state"] = "owner_initializing"
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
        owner_identity = _write_shard_owner_marker(
            owner_marker,
            owner_initializing_marker,
            owner_payload,
            transaction_lock,
        )
        transaction["owner_identity"] = _identity_payload(owner_identity)
        transaction["state"] = "building"
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
        lock_anchor = _read_lock_anchor(transaction_lock)
        lock_anchor["owner"] = None
        _write_lock_anchor(transaction_lock, lock_anchor)
        tracker = _ResourceTracker(work, max_rows_per_run, merge_read_rows)
        assert work is not None and tracker is not None
        membership_snapshot = work / "membership.snapshot.parquet"
        _snapshot_verified_membership(
            split_plan.membership_path,
            membership_snapshot,
            str(split_manifest["membership"]["sha256"]),
        )
        source_runs = _write_source_runs(
            chunks,
            work,
            selected_features=materialized,
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
            membership_snapshot,
            work,
            schema=working_schema,
            max_rows_per_run=max_rows_per_run,
            merge_read_rows=merge_read_rows,
            held_out_attacks=held_out_attacks,
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
            selected_features=selected,
            materialized_features=materialized,
            boolean_fast_path=boolean_contract,
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
            assert journal_identity is not None
            journal_identity = _prepare_owned_shard_work_cleanup(
                transaction, journal, journal_identity, transaction_lock
            )
            journal_identity = _cleanup_owned_shard_work(
                transaction, journal, journal_identity, transaction_lock
            )
            _remove_shard_transaction(journal, journal_identity, transaction_lock)
            transaction = None
            journal_identity = None
            existing = verify_shard_manifest(
                manifest_path,
                split_plan=split_plan,
                preprocessing_fingerprint=preprocessing_fingerprint,
                max_rows_per_run=max_rows_per_run,
                merge_fan_in=merge_fan_in,
                merge_read_rows=merge_read_rows,
            )
            if canonical_json_bytes(
                _manifest_semantics(existing)
            ) != canonical_json_bytes(_manifest_semantics(manifest)):
                raise RuntimeError("immutable shard output semantic conflict")
            result = _plan_from_manifest(manifest_path, existing)
            _release_shard_transaction_lock(transaction_lock)
            lock_released = True
            return result
        final_dataset_root = output / f"dataset={dataset}"
        if final_dataset_root.exists() and (
            final_dataset_root.is_symlink()
            or not final_dataset_root.is_dir()
            or any(final_dataset_root.rglob("*"))
        ):
            raise RuntimeError(
                "incomplete or unsafe immutable shard output already exists"
            )
        artifacts: list[dict[str, Any]] = []
        for entry in entries:
            relative = PurePosixPath(str(entry["path"]))
            staged = staging.joinpath(*relative.parts)
            identity = file_identity(staged)
            if (
                identity.size != int(entry["byte_size"])
                or _sha256_file(staged) != entry["sha256"]
            ):
                raise RuntimeError(f"staged shard changed before publication: {staged}")
            artifacts.append(
                {
                    "relative": relative.as_posix(),
                    "identity": _identity_payload(identity),
                    "size": identity.size,
                    "sha256": str(entry["sha256"]),
                }
            )
        transaction["state"] = "publishing"
        transaction["artifacts"] = artifacts
        transaction["published_prefix"] = 0
        transaction["published_directories"] = []
        transaction["directory_intent"] = None
        transaction["rollback_removed"] = []
        transaction["rollback_intent"] = None
        transaction["directory_rollback_removed"] = []
        transaction["directory_rollback_intent"] = None
        assert journal_identity is not None
        journal_identity = _replace_shard_transaction(
            journal, transaction, journal_identity, transaction_lock
        )
        for entry in entries:
            relative = PurePosixPath(str(entry["path"]))
            destination = output.joinpath(*relative.parts)
            _, journal_identity = _ensure_durable_public_directories(
                output,
                destination.parent,
                transaction,
                journal,
                journal_identity,
                transaction_lock,
            )
        for index, entry in enumerate(entries):
            relative = PurePosixPath(str(entry["path"]))
            staged = staging.joinpath(*relative.parts)
            destination = output.joinpath(*relative.parts)
            _publish_file_no_replace(staged, destination, publication_index=index)
            transaction["published_prefix"] = index + 1
            journal_identity = _replace_shard_transaction(
                journal, transaction, journal_identity, transaction_lock
            )
        journal_identity = _prepare_owned_shard_work_cleanup(
            transaction, journal, journal_identity, transaction_lock
        )
        journal_identity = _cleanup_owned_shard_work(
            transaction, journal, journal_identity, transaction_lock
        )
        _write_manifest_no_replace(manifest_path, manifest, transaction_lock)
        _remove_shard_transaction(journal, journal_identity, transaction_lock)
        transaction = None
        journal_identity = None
        result = _plan_from_manifest(manifest_path, manifest)
        _release_shard_transaction_lock(transaction_lock)
        lock_released = True
        return result
    except BaseException as primary:
        cleanup: list[BaseException] = []
        preserve_committed_publication = False
        publication_rollback_complete = False
        if transaction is not None:
            try:
                _recover_anchored_manifest_temporary(manifest_path, transaction_lock)
                recovery_anchor = _read_lock_anchor(transaction_lock)
                manifest_anchor = recovery_anchor.get("manifest")
                journal_anchor = recovery_anchor.get("journal")
                preserve_committed_publication = (
                    isinstance(manifest_anchor, dict)
                    and manifest_anchor.get("phase") in {"published", "committed"}
                ) or (
                    isinstance(journal_anchor, dict)
                    and journal_anchor.get("phase") == "retiring"
                )
                if (
                    isinstance(journal_anchor, dict)
                    and journal_anchor.get("phase") == "retiring"
                ):
                    if _complete_journal_retirement(journal, transaction_lock):
                        transaction = None
                        journal_identity = None
                elif _path_entry_exists(journal):
                    transaction, journal_identity = _load_shard_transaction(
                        journal,
                        output=output,
                        request_fingerprint=request_fingerprint,
                        dataset=dataset,
                        transaction_lock=transaction_lock,
                    )
                    if preserve_committed_publication:
                        _validate_manifest_commit(transaction, transaction_lock)
                        _remove_shard_transaction(
                            journal, journal_identity, transaction_lock
                        )
                        transaction = None
                        journal_identity = None
                    else:
                        _normalize_transaction_anchor(
                            journal,
                            transaction,
                            journal_identity,
                            transaction_lock,
                        )
            except BaseException as error:
                cleanup.append(error)
        if (
            not preserve_committed_publication
            and transaction is not None
            and transaction.get("state") == "publishing"
            and journal_identity is not None
        ):
            try:
                journal_identity = _validate_publication_and_rollback(
                    transaction,
                    journal,
                    journal_identity,
                    transaction_lock,
                )
                publication_rollback_complete = True
            except BaseException as error:
                cleanup.append(error)
        owned_cleanup_complete = transaction is None
        if transaction is not None and not preserve_committed_publication:
            try:
                if journal_identity is None:
                    raise RuntimeError(
                        "owned shard transaction identity is unavailable"
                    )
                journal_identity = _prepare_owned_shard_work_cleanup(
                    transaction, journal, journal_identity, transaction_lock
                )
                journal_identity = _cleanup_owned_shard_work(
                    transaction, journal, journal_identity, transaction_lock
                )
                owned_cleanup_complete = True
            except BaseException as error:
                cleanup.append(error)
        if (
            publication_rollback_complete
            and owned_cleanup_complete
            and transaction is not None
            and transaction.get("state") == "publishing"
            and journal_identity is not None
        ):
            try:
                journal_identity = _mark_publication_rolled_back(
                    transaction, journal, journal_identity, transaction_lock
                )
            except BaseException as error:
                owned_cleanup_complete = False
                cleanup.append(error)
        if (
            owned_cleanup_complete
            and not preserve_committed_publication
            and (transaction is None or transaction.get("state") != "publishing")
            and journal_identity is not None
            and _path_entry_exists(journal)
        ):
            try:
                _remove_shard_transaction(journal, journal_identity, transaction_lock)
            except BaseException as error:
                cleanup.append(error)
        if not lock_released:
            try:
                _release_shard_transaction_lock(transaction_lock)
                lock_released = True
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
