from __future__ import annotations

import ctypes
import functools
import heapq
import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import tempfile
import warnings
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Iterator, Sequence

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
    file_id: int
    path: Path
    relative_path: str
    logical_identity: str
    fingerprint: FileFingerprint
    snapshot: _VerifiedSnapshot
    row_count: int
    selected_row_count: int
    raw_columns: tuple[str, ...]
    schema: _Schema
    timestamp: _TimestampPlan | None
    numeric_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _IterationPlan:
    source: _SourceSpec
    files: tuple[_FilePlan, ...]
    selection: _SelectionIndex | None
    snapshots: _SnapshotStore
    class_limited: bool
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


@dataclass(frozen=True, slots=True)
class _VerifiedSnapshot:
    path: Path
    pinned: _PinnedStat
    sha256: str


class _HashingReader:
    def __init__(
        self,
        handle: BinaryIO,
        *,
        mirror: BinaryIO | None = None,
        origin: str = "source",
    ) -> None:
        self._handle = handle
        self._mirror = mirror
        self._digest = hashlib.sha256()
        self.bytes_read = 0
        self.origin = origin

    def read(self, size: int = -1) -> bytes:
        block = self._handle.read(size)
        if self._mirror is not None:
            self._mirror.write(block)
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
        _change_time_ns(path, path_stat),
    )


class _WindowsFileBasicInfo(ctypes.Structure):
    _fields_ = [
        ("creation_time", ctypes.c_longlong),
        ("last_access_time", ctypes.c_longlong),
        ("last_write_time", ctypes.c_longlong),
        ("change_time", ctypes.c_longlong),
        ("file_attributes", wintypes.DWORD),
    ]


def _change_time_ns(path: Path, path_stat: os.stat_result) -> int:
    if os.name != "nt":
        return int(path_stat.st_ctime_ns)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        str(path.resolve()),
        0x80,
        0x1 | 0x2 | 0x4,
        None,
        3,
        0x00200000,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise RuntimeError("could not pin dataset source change time")
    try:
        information = _WindowsFileBasicInfo()
        if not get_information(
            handle, 0, ctypes.byref(information), ctypes.sizeof(information)
        ):
            raise RuntimeError("could not pin dataset source change time")
        return int(information.change_time) * 100
    finally:
        close_handle(handle)


def _assert_source_stat(
    path: Path,
    expected: _PinnedStat | FileFingerprint,
    actual: os.stat_result,
    phase: str,
    *,
    check_ctime: bool = True,
) -> None:
    identity = [
        int(actual.st_dev),
        int(actual.st_ino),
        int(actual.st_mode),
        int(actual.st_size),
        int(actual.st_mtime_ns),
    ]
    pinned = [
        expected.device,
        expected.inode,
        expected.mode,
        expected.byte_size,
        expected.mtime_ns,
    ]
    if check_ctime:
        identity.append(_change_time_ns(path, actual))
        pinned.append(expected.ctime_ns)
    if identity != pinned:
        raise RuntimeError(f"dataset source changed during {phase}: {path}")


def _new_snapshot_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="bitguard-verified-source-"))


@functools.lru_cache(maxsize=1)
def _windows_current_sid() -> str:
    identity = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
        timeout=10,
    )
    sid_match = re.search(rb"S-\d+(?:-\d+)+", identity.stdout)
    if sid_match is None:
        raise RuntimeError
    return sid_match.group(0).decode("ascii")


def _secure_snapshot_root(root: Path) -> None:
    if os.name != "nt":
        os.chmod(root, 0o700)
        if stat.S_IMODE(root.stat().st_mode) != 0o700:
            raise RuntimeError("could not enforce private snapshot permissions")
        return
    acl_path = root / ".acl-verification"
    try:
        sid = _windows_current_sid()
        subprocess.run(
            [
                "icacls",
                str(root),
                "/inheritance:r",
                "/remove:g",
                "*S-1-5-18",
                "*S-1-5-32-544",
                "*S-1-3-4",
                "/grant:r",
                f"*{sid}:(OI)(CI)F",
                "/q",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        subprocess.run(
            ["icacls", str(root), "/save", str(acl_path), "/q"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        acl = acl_path.read_bytes().decode("utf-16le").lstrip("\ufeff")
        descriptor = re.search(r"D:([^\r\n]+)", acl)
        aces = re.findall(r"\(([^()]*)\)", descriptor.group(1) if descriptor else "")
        if descriptor is None or not descriptor.group(1).startswith("P"):
            raise RuntimeError
        if aces != [f"A;OICI;FA;;;{sid}"]:
            raise RuntimeError
    except (OSError, subprocess.SubprocessError, UnicodeError, RuntimeError) as exc:
        raise RuntimeError("could not enforce private snapshot permissions") from exc
    finally:
        acl_path.unlink(missing_ok=True)


class _SnapshotBuilder:
    def __init__(self, store: _SnapshotStore, file_id: int) -> None:
        self._store = store
        self.partial_path = store.root / f"{file_id:08d}.partial"
        self.final_path = store.root / f"{file_id:08d}.verified.csv"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        descriptor = os.open(self.partial_path, flags, 0o600)
        self.handle: BinaryIO = os.fdopen(descriptor, "wb", buffering=0)
        self._finished = False
        if os.name != "nt" and stat.S_IMODE(self.partial_path.stat().st_mode) != 0o600:
            self.abort()
            raise RuntimeError("could not enforce private snapshot permissions")

    def commit(self, sha256: str, byte_size: int) -> _VerifiedSnapshot:
        if self._finished:
            raise RuntimeError("snapshot builder is already closed")
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            if os.fstat(self.handle.fileno()).st_size != byte_size:
                raise RuntimeError("verified snapshot size mismatch")
            self.handle.close()
            os.replace(self.partial_path, self.final_path)
            os.chmod(self.final_path, 0o400)
            pinned = _pinned_stat(self.final_path)
            if pinned.byte_size != byte_size:
                raise RuntimeError("verified snapshot size mismatch")
            snapshot = _VerifiedSnapshot(self.final_path, pinned, sha256)
            self._store._publish(self, snapshot)
            self._finished = True
            if os.name != "nt":
                directory = os.open(self._store.root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return snapshot
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._finished:
            return
        self._finished = True
        if not self.handle.closed:
            self.handle.close()
        for path in (self.partial_path, self.final_path):
            if path.exists():
                os.chmod(path, 0o600)
                path.unlink()
        self._store._discard(self)


class _SnapshotStore:
    def __init__(self) -> None:
        self.root = _new_snapshot_root()
        self._builders: set[_SnapshotBuilder] = set()
        self._snapshots: list[_VerifiedSnapshot] = []
        self._closed = False
        try:
            if self.root.exists():
                root_stat = self.root.lstat()
                if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(
                    root_stat.st_mode
                ):
                    raise RuntimeError("invalid snapshot storage")
            else:
                self.root.mkdir(mode=0o700)
            _secure_snapshot_root(self.root)
        except BaseException:
            if self.root.is_dir():
                self.root.rmdir()
            raise

    def begin(self, file_id: int) -> _SnapshotBuilder:
        builder = _SnapshotBuilder(self, file_id)
        self._builders.add(builder)
        return builder

    def _publish(
        self, builder: _SnapshotBuilder, snapshot: _VerifiedSnapshot
    ) -> None:
        self._builders.discard(builder)
        self._snapshots.append(snapshot)

    def _discard(self, builder: _SnapshotBuilder) -> None:
        self._builders.discard(builder)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for builder in tuple(self._builders):
            builder.abort()
        for snapshot in self._snapshots:
            if snapshot.path.exists():
                os.chmod(snapshot.path, 0o600)
                snapshot.path.unlink()
        self._snapshots.clear()
        self.root.rmdir()


class _VerifiedCsvPass:
    def __init__(
        self,
        path: Path,
        pinned: _PinnedStat | FileFingerprint,
        *,
        chunk_size: int,
        phase: str,
        expected_sha256: str | None,
        snapshot_builder: _SnapshotBuilder | None = None,
        source_guard: tuple[Path, FileFingerprint] | None = None,
        origin: str = "source",
    ) -> None:
        self.path = path
        self.pinned = pinned
        self.chunk_size = chunk_size
        self.phase = phase
        self.expected_sha256 = expected_sha256
        self.snapshot_builder = snapshot_builder
        self.source_guard = source_guard
        self.origin = origin
        self.fingerprint: FileFingerprint | None = None
        self.snapshot: _VerifiedSnapshot | None = None

    def _assert_guard(self) -> None:
        if self.source_guard is None:
            return
        path, fingerprint = self.source_guard
        _assert_source_stat(path, fingerprint, path.lstat(), self.phase)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        completed = False
        self._assert_guard()
        with self.path.open("rb", buffering=0) as handle:
            _assert_source_stat(
                self.path,
                self.pinned,
                os.fstat(handle.fileno()),
                self.phase,
                check_ctime=False,
            )
            _assert_source_stat(self.path, self.pinned, self.path.lstat(), self.phase)
            hashing_reader = _HashingReader(
                handle,
                mirror=(
                    self.snapshot_builder.handle
                    if self.snapshot_builder is not None
                    else None
                ),
                origin=self.origin,
            )
            reader: Any = None
            try:
                reader = pd.read_csv(
                    hashing_reader,
                    chunksize=self.chunk_size,
                    low_memory=False,
                )
                for chunk in reader:
                    self._assert_guard()
                    _assert_source_stat(
                        self.path,
                        self.pinned,
                        os.fstat(handle.fileno()),
                        self.phase,
                        check_ctime=False,
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
                if reader is not None:
                    _close_reader(reader)
                if not completed:
                    if self.snapshot_builder is not None:
                        self.snapshot_builder.abort()
                else:
                    try:
                        self._assert_guard()
                        _assert_source_stat(
                            self.path,
                            self.pinned,
                            os.fstat(handle.fileno()),
                            self.phase,
                            check_ctime=False,
                        )
                        _assert_source_stat(
                            self.path,
                            self.pinned,
                            self.path.lstat(),
                            self.phase,
                        )
                        if hashing_reader.bytes_read != self.pinned.byte_size:
                            raise RuntimeError(
                                "dataset source changed during "
                                f"{self.phase}: {self.path}"
                            )
                        digest = hashing_reader.sha256
                        if (
                            self.expected_sha256 is not None
                            and digest != self.expected_sha256
                        ):
                            raise RuntimeError(
                                "dataset source changed during "
                                f"{self.phase}: {self.path}"
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
                        if self.snapshot_builder is not None:
                            self.snapshot = self.snapshot_builder.commit(
                                digest, hashing_reader.bytes_read
                            )
                    except BaseException:
                        if self.snapshot_builder is not None:
                            self.snapshot_builder.abort()
                        raise


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


def _new_selection_path() -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix="bitguard-selection-", suffix=".sqlite3"
    )
    os.close(descriptor)
    return Path(name)


class _SelectionIndex:
    """Disk-backed row membership and materialization-order index."""

    def __init__(self) -> None:
        self.path = _new_selection_path()
        self._connection = sqlite3.connect(self.path)
        self._closed = False
        self._connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=OFF;
            CREATE TABLE file_mode (
                file_id INTEGER PRIMARY KEY,
                all_rows INTEGER NOT NULL
            );
            CREATE TABLE file_row (
                file_id INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                numeric_columns TEXT NOT NULL,
                timestamp_present INTEGER NOT NULL,
                timestamp_numeric INTEGER NOT NULL,
                timestamp_value TEXT,
                PRIMARY KEY (file_id, row_index)
            ) WITHOUT ROWID;
            CREATE TABLE selected_file_row (
                file_id INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                PRIMARY KEY (file_id, row_index)
            ) WITHOUT ROWID;
            CREATE TABLE class_candidate (
                uid TEXT PRIMARY KEY,
                file_id INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                label TEXT NOT NULL,
                numeric_columns TEXT NOT NULL
            );
            CREATE INDEX class_candidate_group
                ON class_candidate(file_id, label, row_index);
            CREATE TABLE selected_class_row (
                uid TEXT PRIMARY KEY,
                materialization_order INTEGER NOT NULL
            );
            """
        )

    def add_file_facts(
        self,
        facts: Iterable[tuple[int, int, str, int, int, str | None]],
    ) -> None:
        self._connection.executemany(
            "INSERT INTO file_row VALUES (?, ?, ?, ?, ?, ?)", facts
        )

    def select_all_file_rows(self, file_id: int) -> None:
        self._connection.execute(
            "INSERT INTO file_mode(file_id, all_rows) VALUES (?, 1)",
            (file_id,),
        )
        self._connection.commit()

    def select_file_rows(self, file_id: int, rows: Iterable[int]) -> None:
        self._connection.execute(
            "INSERT INTO file_mode(file_id, all_rows) VALUES (?, 0)",
            (file_id,),
        )
        self._connection.executemany(
            "INSERT INTO selected_file_row VALUES (?, ?)",
            ((file_id, int(row)) for row in rows),
        )
        self._connection.commit()

    def selected_file_mask(
        self, file_id: int, positions: Sequence[int]
    ) -> np.ndarray:
        if not positions:
            return np.zeros(0, dtype=bool)
        mode = self._connection.execute(
            "SELECT all_rows FROM file_mode WHERE file_id = ?", (file_id,)
        ).fetchone()
        if mode is None or bool(mode[0]):
            return np.ones(len(positions), dtype=bool)
        selected = {
            int(row[0])
            for row in self._connection.execute(
                "SELECT row_index FROM selected_file_row "
                "WHERE file_id = ? AND row_index BETWEEN ? AND ?",
                (file_id, int(positions[0]), int(positions[-1])),
            )
        }
        return np.fromiter(
            (int(position) in selected for position in positions),
            dtype=bool,
            count=len(positions),
        )

    def selected_facts(
        self, file_id: int
    ) -> Iterator[tuple[str, int, int, str | None]]:
        mode = self._connection.execute(
            "SELECT all_rows FROM file_mode WHERE file_id = ?", (file_id,)
        ).fetchone()
        if mode is None or bool(mode[0]):
            query = (
                "SELECT numeric_columns, timestamp_present, "
                "timestamp_numeric, timestamp_value FROM file_row "
                "WHERE file_id = ? ORDER BY row_index"
            )
        else:
            query = (
                "SELECT f.numeric_columns, f.timestamp_present, "
                "f.timestamp_numeric, f.timestamp_value FROM file_row AS f "
                "JOIN selected_file_row AS s USING(file_id, row_index) "
                "WHERE f.file_id = ? ORDER BY f.row_index"
            )
        yield from self._connection.execute(query, (file_id,))

    def discard_file_facts(self, file_id: int) -> None:
        self._connection.execute(
            "DELETE FROM file_row WHERE file_id = ?", (file_id,)
        )
        self._connection.commit()

    def add_class_candidates(
        self, rows: Iterable[tuple[str, int, int, str, str]]
    ) -> None:
        self._connection.executemany(
            "INSERT INTO class_candidate VALUES (?, ?, ?, ?, ?)", rows
        )
        self._connection.commit()

    def iter_class_candidates(
        self, file_id: int, label: str
    ) -> Iterator[tuple[str, str]]:
        yield from self._connection.execute(
            "SELECT uid, numeric_columns FROM class_candidate "
            "WHERE file_id = ? AND label = ? ORDER BY row_index",
            (file_id, label),
        )

    def set_selected_class_rows(self, ordered_uids: Iterable[str]) -> None:
        self._connection.executemany(
            "INSERT INTO selected_class_row VALUES (?, ?)",
            ((uid, ordinal) for ordinal, uid in enumerate(ordered_uids)),
        )
        self._connection.commit()

    def selected_class_mask(self, uids: Sequence[str]) -> np.ndarray:
        selected: set[str] = set()
        for start in range(0, len(uids), 500):
            batch = list(uids[start : start + 500])
            placeholders = ",".join("?" for _ in batch)
            selected.update(
                str(row[0])
                for row in self._connection.execute(
                    f"SELECT uid FROM selected_class_row WHERE uid IN ({placeholders})",
                    batch,
                )
            )
        return np.fromiter(
            (str(uid) in selected for uid in uids),
            dtype=bool,
            count=len(uids),
        )

    def selected_class_facts(self) -> Iterator[tuple[int, str]]:
        yield from self._connection.execute(
            "SELECT c.file_id, c.numeric_columns FROM class_candidate AS c "
            "JOIN selected_class_row AS s USING(uid) "
            "ORDER BY c.file_id, c.row_index"
        )

    def materialization_order(self, uids: Sequence[str]) -> dict[str, int]:
        order: dict[str, int] = {}
        for start in range(0, len(uids), 500):
            batch = list(uids[start : start + 500])
            placeholders = ",".join("?" for _ in batch)
            order.update(
                (str(uid), int(position))
                for uid, position in self._connection.execute(
                    "SELECT uid, materialization_order FROM selected_class_row "
                    f"WHERE uid IN ({placeholders})",
                    batch,
                )
            )
        return order

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()
        self.path.unlink(missing_ok=True)
        Path(f"{self.path}-journal").unlink(missing_ok=True)
        Path(f"{self.path}-wal").unlink(missing_ok=True)
        Path(f"{self.path}-shm").unlink(missing_ok=True)


def _iter_selected_raw(
    plan: _FilePlan,
    chunk_size: int,
    selection: _SelectionIndex | None,
    *,
    include_empty: bool = False,
) -> Iterator[tuple[int, pd.DataFrame]]:
    offset = 0
    csv_pass = _VerifiedCsvPass(
        plan.snapshot.path,
        plan.snapshot.pinned,
        chunk_size=chunk_size,
        phase="CSV normalization pass",
        expected_sha256=plan.snapshot.sha256,
        source_guard=(plan.path, plan.fingerprint),
        origin="snapshot",
    )
    for chunk in csv_pass:
        source_row_start = offset
        positions = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        frame = chunk.copy()
        frame["__source_row_index"] = positions
        if selection is not None:
            mask = selection.selected_file_mask(
                plan.file_id, positions.astype(int).tolist()
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


def _raw_feature_columns(
    kind: str,
    schema: _Schema,
    raw_columns: Sequence[str],
    drop_columns: set[str],
) -> tuple[str, ...]:
    metadata_sources: set[str] = set()
    if kind != "nbaiot":
        metadata_sources = {
            column
            for column in (
                schema.label,
                schema.raw_attack,
                schema.device,
                schema.timestamp,
            )
            if column is not None
        }
    excluded = META_COLUMNS | drop_columns | metadata_sources
    return tuple(column for column in raw_columns if column not in excluded)


def _timestamp_plan(
    schema: _Schema,
    *,
    numeric_count: int,
    timestamp_count: int,
    first_timestamp: str | None,
) -> _TimestampPlan | None:
    if schema.timestamp is None:
        return None
    numeric_mode = timestamp_count > 0 and numeric_count / timestamp_count >= 0.95
    datetime_format: str | None = None
    if not numeric_mode and first_timestamp is not None:
        datetime_format = guess_datetime_format(first_timestamp)
        if datetime_format is None:
            datetime_format = "mixed"
    return _TimestampPlan(numeric_mode, datetime_format)


def _selected_fact_summary(
    facts: Iterable[tuple[str, int, int, str | None]],
) -> tuple[tuple[str, ...], int, int, str | None]:
    numeric_order: list[str] = []
    numeric_seen: set[str] = set()
    timestamp_count = 0
    timestamp_numeric = 0
    first_timestamp: str | None = None
    for encoded, present, numeric, value in facts:
        for column in json.loads(encoded):
            if column not in numeric_seen:
                numeric_seen.add(column)
                numeric_order.append(str(column))
        timestamp_count += int(present)
        timestamp_numeric += int(numeric)
        if first_timestamp is None and value is not None:
            first_timestamp = str(value)
    return (
        tuple(numeric_order),
        timestamp_numeric,
        timestamp_count,
        first_timestamp,
    )


def _plan_file(
    *,
    file_id: int,
    path: Path,
    relative_path: str,
    source: _SourceSpec,
    cfg: dict[str, Any],
    chunk_size: int,
    max_rows: int | None,
    seed: int,
    budget: _SourceRowBudget,
    selection: _SelectionIndex | None,
    snapshots: _SnapshotStore,
    drop_columns: set[str],
) -> _FilePlan:
    pinned = _pinned_stat(path)
    schema: _Schema | None = None
    raw_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()
    numeric_seen: set[str] = set()
    numeric_order: list[str] = []
    timestamp_count = 0
    timestamp_numeric = 0
    first_timestamp: str | None = None
    offset = 0
    snapshot_builder = snapshots.begin(file_id)
    csv_pass = _VerifiedCsvPass(
        path,
        pinned,
        chunk_size=chunk_size,
        phase="source planning",
        expected_sha256=None,
        snapshot_builder=snapshot_builder,
    )
    for chunk in csv_pass:
        if schema is None:
            schema = _schema_for(source.kind, chunk, cfg, path)
            raw_columns = tuple(str(column) for column in chunk.columns)
            feature_columns = _raw_feature_columns(
                source.kind, schema, raw_columns, drop_columns
            )
        rows = len(chunk)
        budget.consume(rows)
        if rows == 0:
            continue
        positions = range(offset, offset + rows)
        offset += rows
        numeric_by_row: list[list[str]] = [[] for _ in range(rows)]
        for column in feature_columns:
            mask = pd.to_numeric(chunk[column], errors="coerce").notna().to_numpy()
            if bool(mask.any()) and max_rows is None and column not in numeric_seen:
                numeric_seen.add(column)
                numeric_order.append(column)
            if max_rows is not None:
                for row_offset in np.flatnonzero(mask):
                    numeric_by_row[int(row_offset)].append(column)
        timestamp_present = np.zeros(rows, dtype=np.int8)
        timestamp_is_numeric = np.zeros(rows, dtype=np.int8)
        timestamp_values: list[str | None] = [None] * rows
        if schema.timestamp is not None:
            values = chunk[schema.timestamp]
            converted = pd.to_numeric(values, errors="coerce")
            timestamp_present.fill(1)
            timestamp_is_numeric = converted.notna().to_numpy(dtype=np.int8)
            non_missing = values.notna().to_numpy()
            for row_offset in np.flatnonzero(non_missing):
                timestamp_values[int(row_offset)] = str(values.iloc[int(row_offset)])
            if max_rows is None:
                timestamp_count += rows
                timestamp_numeric += int(timestamp_is_numeric.sum())
                if first_timestamp is None:
                    first = next(
                        (value for value in timestamp_values if value is not None),
                        None,
                    )
                    first_timestamp = first
        if max_rows is not None:
            assert selection is not None
            selection.add_file_facts(
                (
                    file_id,
                    int(row_index),
                    json.dumps(numeric_by_row[row_offset], separators=(",", ":")),
                    int(timestamp_present[row_offset]),
                    int(timestamp_is_numeric[row_offset]),
                    timestamp_values[row_offset],
                )
                for row_offset, row_index in enumerate(positions)
            )
    fingerprint = csv_pass.fingerprint
    snapshot = csv_pass.snapshot
    assert fingerprint is not None
    assert snapshot is not None
    assert schema is not None
    identity = logical_source_id(source.kind, relative_path, fingerprint.sha256)
    selected_count = offset
    if selection is not None:
        if max_rows is None or offset <= max_rows:
            selection.select_all_file_rows(file_id)
        else:
            retained: list[tuple[int, int]] = []
            for source_index in range(offset):
                key = source_sampling_key(seed, identity, source_index)
                entry = (-key, -source_index)
                if len(retained) < max_rows:
                    heapq.heappush(retained, entry)
                elif entry > retained[0]:
                    heapq.heapreplace(retained, entry)
            selection.select_file_rows(
                file_id, (-source_index for _, source_index in retained)
            )
            selected_count = max_rows
    if max_rows is not None:
        assert selection is not None
        (
            selected_numeric,
            timestamp_numeric,
            timestamp_count,
            first_timestamp,
        ) = _selected_fact_summary(selection.selected_facts(file_id))
        numeric_seen = set(selected_numeric)
        numeric_order = [
            column for column in feature_columns if column in numeric_seen
        ]
        selection.discard_file_facts(file_id)
    else:
        numeric_order = [
            column for column in feature_columns if column in numeric_seen
        ]
    timestamp = _timestamp_plan(
        schema,
        numeric_count=timestamp_numeric,
        timestamp_count=timestamp_count,
        first_timestamp=first_timestamp,
    )
    return _FilePlan(
        file_id=file_id,
        path=path,
        relative_path=relative_path,
        logical_identity=identity,
        fingerprint=fingerprint,
        snapshot=snapshot,
        row_count=offset,
        selected_row_count=selected_count,
        raw_columns=raw_columns,
        schema=schema,
        timestamp=timestamp,
        numeric_columns=tuple(numeric_order),
    )


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
    plan: _FilePlan,
    source: _SourceSpec,
    chunk_size: int,
    selection: _SelectionIndex | None,
) -> Iterator[NormalizedChunk]:
    for source_row_start, frame in _iter_selected_raw(
        plan, chunk_size, selection
    ):
        yield NormalizedChunk(
            _normalize_frame(frame, plan, source),
            plan.relative_path,
            source_row_start,
        )


def _ordered_feature_columns(
    files: tuple[_FilePlan, ...],
    selected_file_ids: set[int],
    numeric_columns: set[str],
    source_kind: str,
    drop_columns: set[str],
) -> tuple[str, ...]:
    column_order: list[str] = []
    seen: set[str] = set()
    for plan in files:
        if plan.file_id not in selected_file_ids:
            continue
        for column in _raw_feature_columns(
            source_kind, plan.schema, plan.raw_columns, drop_columns
        ):
            if column not in seen:
                seen.add(column)
                column_order.append(column)
    features = tuple(
        column for column in column_order if column in numeric_columns
    )
    if not features:
        raise ValueError("no numeric feature columns were found")
    return features


def _select_class_rows(
    files: tuple[_FilePlan, ...],
    source: _SourceSpec,
    selection: _SelectionIndex,
    *,
    chunk_size: int,
    limit: int,
    seed: int,
    drop_columns: set[str],
) -> tuple[str, ...]:
    rng = np.random.default_rng(seed)
    heaps: dict[str, list[tuple[float, int, str]]] = {}
    ordinals: dict[str, int] = {}
    for plan in files:
        label_order: list[str] = []
        seen: set[str] = set()
        candidates: list[tuple[str, int, int, str, str]] = []
        feature_columns = _raw_feature_columns(
            source.kind, plan.schema, plan.raw_columns, drop_columns
        )
        for chunk in _iter_file_normalized(
            plan, source, chunk_size, selection
        ):
            frame = chunk.frame
            numeric_by_row: list[list[str]] = [[] for _ in range(len(frame))]
            for column in feature_columns:
                mask = pd.to_numeric(frame[column], errors="coerce").notna().to_numpy()
                for row_offset in np.flatnonzero(mask):
                    numeric_by_row[int(row_offset)].append(column)
            labels = frame["behavior_label"].astype(str).tolist()
            for label in labels:
                if label not in seen:
                    seen.add(label)
                    label_order.append(label)
            candidates.extend(
                (
                    str(uid),
                    plan.file_id,
                    int(row_index),
                    label,
                    json.dumps(
                        numeric_by_row[row_offset], separators=(",", ":")
                    ),
                )
                for row_offset, (uid, row_index, label) in enumerate(
                    zip(
                        frame["row_uid"].astype(str),
                        frame["sequence_index"].astype(int),
                        labels,
                    )
                )
            )
            if len(candidates) >= 1_000:
                selection.add_class_candidates(candidates)
                candidates.clear()
        if candidates:
            selection.add_class_candidates(candidates)
        for label in label_order:
            batch: list[tuple[str, str]] = []
            for candidate in selection.iter_class_candidates(plan.file_id, label):
                batch.append(candidate)
                if len(batch) < chunk_size:
                    continue
                keys = rng.random(len(batch))
                heap = heaps.setdefault(label, [])
                ordinal = ordinals.get(label, 0)
                for key, (uid, _) in zip(keys, batch):
                    entry = (-float(key), -ordinal, uid)
                    ordinal += 1
                    if len(heap) < limit:
                        heapq.heappush(heap, entry)
                    elif entry > heap[0]:
                        heapq.heapreplace(heap, entry)
                ordinals[label] = ordinal
                batch.clear()
            if batch:
                keys = rng.random(len(batch))
                heap = heaps.setdefault(label, [])
                ordinal = ordinals.get(label, 0)
                for key, (uid, _) in zip(keys, batch):
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
    if not ordered_uids:
        raise ValueError("dataset contains no rows")
    selection.set_selected_class_rows(ordered_uids)
    selected_files: set[int] = set()
    numeric_columns: set[str] = set()
    for file_id, encoded in selection.selected_class_facts():
        selected_files.add(int(file_id))
        numeric_columns.update(str(column) for column in json.loads(encoded))
    return _ordered_feature_columns(
        files,
        selected_files,
        numeric_columns,
        source.kind,
        drop_columns,
    )


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
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive or null")
    snapshots = _SnapshotStore()
    selection: _SelectionIndex | None = None
    file_plans: list[_FilePlan] = []
    drop_columns = set(cfg.get("drop_columns", []))
    try:
        selection = (
            _SelectionIndex()
            if max_rows is not None or class_limit is not None
            else None
        )
        for file_id, (path, relative_path) in enumerate(
            zip(source.files, source.relative_paths)
        ):
            file_plans.append(
                _plan_file(
                    file_id=file_id,
                    path=path,
                    relative_path=relative_path,
                    source=source,
                    cfg=cfg,
                    chunk_size=chunk_size,
                    max_rows=max_rows,
                    seed=int(config["experiment"]["seed"]),
                    budget=budget,
                    selection=selection,
                    snapshots=snapshots,
                    drop_columns=drop_columns,
                )
            )
        files = tuple(file_plans)
        selected_files = {
            plan.file_id for plan in files if plan.selected_row_count > 0
        }
        if class_limit is not None:
            if class_limit <= 0:
                raise ValueError("max_rows_per_class must be positive or null")
            if not selected_files:
                raise ValueError("dataset contains no rows")
            assert selection is not None
            features = _select_class_rows(
                files,
                source,
                selection,
                chunk_size=chunk_size,
                limit=class_limit,
                seed=int(config["experiment"]["seed"]),
                drop_columns=drop_columns,
            )
        else:
            numeric_columns = {
                column for plan in files for column in plan.numeric_columns
            }
            features = _ordered_feature_columns(
                files,
                selected_files,
                numeric_columns,
                source.kind,
                drop_columns,
            )
        return _IterationPlan(
            source=source,
            files=files,
            selection=selection,
            snapshots=snapshots,
            class_limited=class_limit is not None,
            feature_columns=features,
        )
    except BaseException:
        try:
            if selection is not None:
                selection.close()
        finally:
            snapshots.close()
        raise


def _close_iteration_plan(plan: _IterationPlan) -> None:
    try:
        if plan.selection is not None:
            plan.selection.close()
    finally:
        plan.snapshots.close()


def _iter_planned_file_chunks(
    config: dict[str, Any], plan: _IterationPlan, file_plan: _FilePlan
) -> Iterator[NormalizedChunk]:
    chunk_size = int(config["dataset"]["chunk_size"])
    for chunk in _iter_file_normalized(
        file_plan, plan.source, chunk_size, plan.selection
    ):
        frame = chunk.frame
        if plan.class_limited:
            assert plan.selection is not None
            mask = plan.selection.selected_class_mask(
                frame["row_uid"].astype(str).tolist()
            )
            frame = frame.loc[mask].copy()
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
    try:
        yield from _iter_planned_chunks(config, plan)
    finally:
        _close_iteration_plan(plan)


def load_normalized_dataset(
    config: dict[str, Any],
    path_override: Path | None = None,
    *,
    dataset_type: str | None = None,
) -> LoadedDataset:
    """Materialize normalized chunks at the legacy in-memory compatibility boundary."""

    plan = _build_iteration_plan(config, path_override, True, dataset_type)
    try:
        frames: list[pd.DataFrame] = []
        for file_plan in plan.files:
            if file_plan.selected_row_count == 0 and not plan.class_limited:
                header = pd.DataFrame(columns=file_plan.raw_columns)
                frames.append(_normalize_frame(header, file_plan, plan.source))
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
        if plan.class_limited:
            assert plan.selection is not None
            uids = combined["row_uid"].astype(str).tolist()
            order = plan.selection.materialization_order(uids)
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
    finally:
        _close_iteration_plan(plan)
