"""Crash-safe disk-backed calibration state for complete validation splits.

Only the contiguous prefix named by ``journal.json`` is committed.  Range bytes
are flushed and synced before the journal is atomically replaced, so an
interrupted writer can safely overwrite an uncommitted suffix on resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


CACHE_ALGORITHM = "bitguard.calibration-cache.v1"
_LAYOUT_SCHEMA = "bitguard.calibration-cache-layout.v1"
_JOURNAL_SCHEMA = "bitguard.calibration-cache-journal.v1"
_LAYOUT_FILE = "layout.json"
_JOURNAL_FILE = "journal.json"


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """Immutable scientific and physical identity of one calibration cache."""

    prepared_descriptor_fingerprint: str
    shard_fingerprint: str
    preprocessor_fingerprint: str
    source_fingerprint: str
    split: str
    row_count: int
    class_labels: tuple[str, ...]
    selected_features: tuple[str, ...]
    boolean_features: tuple[str, ...]
    device_id_width: int
    source_id_width: int
    algorithm: str = CACHE_ALGORITHM

    def __post_init__(self) -> None:
        for field in (
            "prepared_descriptor_fingerprint",
            "shard_fingerprint",
            "preprocessor_fingerprint",
            "source_fingerprint",
            "split",
            "algorithm",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"CacheLayout.{field} must be a non-empty string")
        for field in ("row_count", "device_id_width", "source_id_width"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"CacheLayout.{field} must be a positive integer")
        for field in ("class_labels", "selected_features", "boolean_features"):
            values = getattr(self, field)
            if not isinstance(values, tuple):
                raise ValueError(f"CacheLayout.{field} must be a tuple")
            if field != "boolean_features" and not values:
                raise ValueError(f"CacheLayout.{field} must not be empty")
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"CacheLayout.{field} contains an invalid name")
            if len(set(values)) != len(values):
                raise ValueError(f"CacheLayout.{field} contains duplicate names")

    def semantic_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["class_labels"] = list(self.class_labels)
        payload["selected_features"] = list(self.selected_features)
        payload["boolean_features"] = list(self.boolean_features)
        return payload

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.semantic_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": _LAYOUT_SCHEMA,
            **self.semantic_dict(),
        }
        payload["fingerprint"] = _fingerprint(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CacheLayout:
        value = dict(payload)
        fingerprint = value.pop("fingerprint", None)
        if (
            value.get("schema_version") != _LAYOUT_SCHEMA
            or fingerprint != _fingerprint(value)
        ):
            raise RuntimeError("calibration cache layout fingerprint mismatch")
        value.pop("schema_version")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise RuntimeError("calibration cache layout schema mismatch")
        for field in ("class_labels", "selected_features", "boolean_features"):
            raw = value[field]
            if not isinstance(raw, list):
                raise RuntimeError("calibration cache layout schema mismatch")
            value[field] = tuple(raw)
        try:
            return cls(**value)
        except (TypeError, ValueError) as error:
            raise RuntimeError("calibration cache layout schema mismatch") from error


@dataclass(frozen=True, slots=True)
class _ArraySpec:
    dtype: str
    shape: tuple[int, ...]

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        return np.dtype(self.dtype)

    @property
    def byte_size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.numpy_dtype.itemsize


def _array_specs(layout: CacheLayout) -> dict[str, _ArraySpec]:
    rows = layout.row_count
    classes = len(layout.class_labels)
    selected = len(layout.selected_features)
    boolean = len(layout.boolean_features)
    specs = {
        "cache_position": _ArraySpec("<i8", (rows,)),
        "uid_digest": _ArraySpec("|u1", (rows, 32)),
        "true_label": _ArraySpec("<i4", (rows,)),
        "known_probabilities": _ArraySpec("<f4", (rows, classes)),
        "selected_values": _ArraySpec("<f4", (rows, selected)),
        "tiny_benign_probability": _ArraySpec("<f4", (rows,)),
        "timestamp": _ArraySpec("<f8", (rows,)),
        "sequence": _ArraySpec("<i8", (rows,)),
        "device_id_bytes": _ArraySpec("|u1", (rows, layout.device_id_width)),
        "device_id_length": _ArraySpec("<u4", (rows,)),
        "source_id_bytes": _ArraySpec("|u1", (rows, layout.source_id_width)),
        "source_id_length": _ArraySpec("<u4", (rows,)),
        "routed_probabilities": _ArraySpec("<f4", (rows, classes)),
        "exit_stage": _ArraySpec("<i2", (rows,)),
        "boolean_flags": _ArraySpec("|u1", (rows, boolean)),
    }
    return specs


class CalibrationCache:
    """A committed-prefix view over preallocated memory-mapped arrays."""

    def __init__(
        self,
        root: Path,
        layout: CacheLayout,
        arrays: dict[str, np.ndarray[Any, Any]],
        committed_rows: int,
        commits: list[dict[str, Any]],
        *,
        readonly: bool,
    ) -> None:
        self.root = root
        self.layout = layout
        self._arrays = arrays
        self._array_view: Mapping[str, np.ndarray[Any, Any]] = MappingProxyType(arrays)
        self._committed_rows = committed_rows
        self._commits = commits
        self._readonly = readonly
        self._closed = False

    @classmethod
    def create(cls, root: Path | str, layout: CacheLayout) -> CalibrationCache:
        path = Path(root).expanduser()
        try:
            path.mkdir(mode=0o700, parents=False, exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(f"calibration cache already exists: {path}") from error
        _validate_directory(path)
        try:
            _write_exclusive(path / _LAYOUT_FILE, _json_bytes(layout.to_dict()))
            for name, spec in _array_specs(layout).items():
                _preallocate(path / f"{name}.bin", spec.byte_size)
            journal = _journal_payload(layout.fingerprint, 0, [])
            _write_exclusive(path / _JOURNAL_FILE, _json_bytes(journal))
            _fsync_directory(path)
            return cls._open(path, layout, readonly=False)
        except BaseException:
            # A failed first creation is intentionally not accepted as resumable.
            # Leave evidence in place for explicit operator cleanup.
            raise

    @classmethod
    def open_readonly(
        cls, root: Path | str, expected_layout: CacheLayout
    ) -> CalibrationCache:
        return cls._open(Path(root).expanduser(), expected_layout, readonly=True)

    @classmethod
    def open_resume(
        cls, root: Path | str, expected_layout: CacheLayout
    ) -> CalibrationCache:
        return cls._open(Path(root).expanduser(), expected_layout, readonly=False)

    @classmethod
    def _open(
        cls, root: Path, expected_layout: CacheLayout, *, readonly: bool
    ) -> CalibrationCache:
        _validate_directory(root)
        stored = CacheLayout.from_dict(_read_json(root / _LAYOUT_FILE, "layout"))
        if stored.fingerprint != expected_layout.fingerprint:
            raise RuntimeError("calibration cache layout does not match expected layout")
        journal = _read_json(root / _JOURNAL_FILE, "journal")
        committed_rows, commits = _validate_journal(journal, stored)
        specs = _array_specs(stored)
        arrays: dict[str, np.ndarray[Any, Any]] = {}
        try:
            for name, spec in specs.items():
                file_path = root / f"{name}.bin"
                _validate_regular_file(file_path, expected_size=spec.byte_size)
                if spec.byte_size == 0:
                    arrays[name] = np.empty(spec.shape, dtype=spec.numpy_dtype)
                    if readonly:
                        arrays[name].setflags(write=False)
                else:
                    arrays[name] = np.memmap(
                        file_path,
                        dtype=spec.numpy_dtype,
                        mode="r" if readonly else "r+",
                        shape=spec.shape,
                        order="C",
                    )
            _verify_committed_ranges(root, specs, commits)
        except BaseException:
            _close_memmaps(arrays.values())
            raise
        return cls(
            root.resolve(), stored, arrays, committed_rows, commits, readonly=readonly
        )

    @property
    def arrays(self) -> Mapping[str, np.ndarray[Any, Any]]:
        self._ensure_open()
        return self._array_view

    @property
    def committed_rows(self) -> int:
        self._ensure_open()
        return self._committed_rows

    @property
    def readonly(self) -> bool:
        self._ensure_open()
        return self._readonly

    def commit_range(self, start: int, values: Mapping[str, object]) -> None:
        """Durably append one fully populated contiguous cache range."""

        self._ensure_open()
        if self._readonly:
            raise RuntimeError("read-only calibration cache cannot be committed")
        if isinstance(start, bool) or not isinstance(start, int) or start != self._committed_rows:
            raise ValueError("cache commit must start at the committed prefix")
        prepared, rows = _prepare_batch(self.layout, start, values)
        end = start + rows
        if end > self.layout.row_count:
            raise ValueError("cache commit exceeds the declared row count")

        for name, array in prepared.items():
            self._arrays[name][start:end] = array
        specs = _array_specs(self.layout)
        for array in self._arrays.values():
            if isinstance(array, np.memmap):
                array.flush()
        for name in specs:
            _fsync_file(self.root / f"{name}.bin")

        commit = {
            "start": start,
            "end": end,
            "sha256": _range_digest(self.root, specs, start, end),
        }
        commits = [*self._commits, commit]
        journal = _journal_payload(self.layout.fingerprint, end, commits)
        _write_journal(self.root, journal)
        self._commits = commits
        self._committed_rows = end

    def read_identifiers(self, field: str, start: int, end: int) -> tuple[str, ...]:
        """Decode a bounded committed range of losslessly stored UTF-8 identifiers."""

        self._ensure_open()
        if field not in {"device_id", "source_id"}:
            raise ValueError("field must be 'device_id' or 'source_id'")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end > self._committed_rows
        ):
            raise ValueError("identifier range is outside the committed prefix")
        raw = self._arrays[f"{field}_bytes"]
        lengths = self._arrays[f"{field}_length"]
        return tuple(
            bytes(raw[index, : int(lengths[index])]).decode("utf-8")
            for index in range(start, end)
        )

    def close(self) -> None:
        if self._closed:
            return
        _close_memmaps(self._arrays.values())
        self._arrays.clear()
        self._closed = True

    def __enter__(self) -> CalibrationCache:
        self._ensure_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("calibration cache is closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def class_weights_from_counts(
    counts: Mapping[str, int], active_labels: Sequence[str]
) -> np.ndarray:
    """Return the existing normalized inverse-frequency class weights."""

    labels = tuple(active_labels)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("active_labels must contain non-empty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("active_labels must not contain duplicates")
    if not isinstance(counts, Mapping):
        raise ValueError("counts must be a mapping")
    values: list[int] = []
    for label in labels:
        if label not in counts:
            raise ValueError(f"missing count for active class {label!r}")
        count = counts[label]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"count for active class {label!r} must be positive integer")
        values.append(count)
    total = sum(values)
    weights = total / (len(values) * np.asarray(values, dtype=np.float64))
    return (weights / weights.mean()).astype(np.float32)


def _require_array(
    values: Mapping[str, object],
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray[Any, Any]:
    value = values[name]
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        raise ValueError(
            f"{name} must be a NumPy array with dtype {dtype} and shape {shape}"
        )
    return value


def _encode_identifiers(
    value: object, *, name: str, rows: int, width: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        raise ValueError(f"{name} must not use object dtype")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    if len(value) != rows:
        raise ValueError(f"{name} length does not match the committed range")
    encoded = np.zeros((rows, width), dtype=np.uint8)
    lengths = np.zeros(rows, dtype=np.uint32)
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{name} must contain only strings")
        payload = item.encode("utf-8")
        if len(payload) > width:
            raise ValueError(f"{name} value exceeds its declared UTF-8 width")
        encoded[index, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        lengths[index] = len(payload)
    return encoded, lengths


def _prepare_batch(
    layout: CacheLayout, start: int, values: Mapping[str, object]
) -> tuple[dict[str, np.ndarray[Any, Any]], int]:
    required = {
        "cache_position",
        "uid_digest",
        "true_label",
        "known_probabilities",
        "selected_values",
        "tiny_benign_probability",
        "boolean_flags",
        "timestamp",
        "sequence",
        "device_id",
        "source_id",
        "routed_probabilities",
        "exit_stage",
    }
    if not isinstance(values, Mapping) or set(values) != required:
        raise ValueError("cache range fields do not match the cache schema")
    positions_value = values["cache_position"]
    if not isinstance(positions_value, np.ndarray) or positions_value.ndim != 1:
        raise ValueError("cache_position must be a one-dimensional NumPy array")
    rows = len(positions_value)
    if rows <= 0:
        raise ValueError("cache commit range must not be empty")
    classes = len(layout.class_labels)
    selected = len(layout.selected_features)
    booleans = len(layout.boolean_features)
    positions = _require_array(values, "cache_position", np.dtype("<i8"), (rows,))
    if not np.array_equal(positions, np.arange(start, start + rows, dtype=np.int64)):
        raise ValueError("cache_position must be the collision-free contiguous range")
    prepared: dict[str, np.ndarray[Any, Any]] = {
        "cache_position": positions,
        "uid_digest": _require_array(values, "uid_digest", np.dtype("|u1"), (rows, 32)),
        "true_label": _require_array(values, "true_label", np.dtype("<i4"), (rows,)),
        "known_probabilities": _require_array(
            values, "known_probabilities", np.dtype("<f4"), (rows, classes)
        ),
        "selected_values": _require_array(
            values, "selected_values", np.dtype("<f4"), (rows, selected)
        ),
        "tiny_benign_probability": _require_array(
            values, "tiny_benign_probability", np.dtype("<f4"), (rows,)
        ),
        "timestamp": _require_array(values, "timestamp", np.dtype("<f8"), (rows,)),
        "sequence": _require_array(values, "sequence", np.dtype("<i8"), (rows,)),
        "routed_probabilities": _require_array(
            values, "routed_probabilities", np.dtype("<f4"), (rows, classes)
        ),
        "exit_stage": _require_array(values, "exit_stage", np.dtype("<i2"), (rows,)),
    }
    boolean_flags = _require_array(
        values, "boolean_flags", np.dtype("bool"), (rows, booleans)
    )
    prepared["boolean_flags"] = boolean_flags.astype(np.uint8, copy=False)
    true_label = prepared["true_label"]
    if np.any(true_label < 0) or np.any(true_label >= classes):
        raise ValueError("true_label contains an index outside class_labels")
    for name in (
        "known_probabilities",
        "selected_values",
        "tiny_benign_probability",
        "timestamp",
        "routed_probabilities",
    ):
        if not np.all(np.isfinite(prepared[name])):
            raise ValueError(f"{name} must contain only finite values")
    for name in (
        "known_probabilities",
        "tiny_benign_probability",
        "routed_probabilities",
    ):
        if np.any(prepared[name] < 0.0) or np.any(prepared[name] > 1.0):
            raise ValueError(f"{name} must contain probabilities in [0, 1]")
    for name in ("known_probabilities", "routed_probabilities"):
        if not np.allclose(
            prepared[name].sum(axis=1, dtype=np.float64), 1.0, rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"{name} rows must sum to one")
    device_bytes, device_lengths = _encode_identifiers(
        values["device_id"], name="device_id", rows=rows, width=layout.device_id_width
    )
    source_bytes, source_lengths = _encode_identifiers(
        values["source_id"], name="source_id", rows=rows, width=layout.source_id_width
    )
    prepared.update(
        {
            "device_id_bytes": device_bytes,
            "device_id_length": device_lengths,
            "source_id_bytes": source_bytes,
            "source_id_length": source_lengths,
        }
    )
    return prepared, rows


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _validate_directory(path: Path) -> None:
    try:
        result = path.lstat()
    except OSError as error:
        raise RuntimeError(f"cannot inspect calibration cache directory: {path}") from error
    if (
        stat.S_ISLNK(result.st_mode)
        or not stat.S_ISDIR(result.st_mode)
        or bool(getattr(result, "st_reparse_tag", 0))
    ):
        raise RuntimeError(f"calibration cache path is not a regular directory: {path}")


def _validate_regular_file(path: Path, *, expected_size: int | None = None) -> None:
    try:
        result = path.lstat()
    except OSError as error:
        raise RuntimeError(f"cannot inspect calibration cache artifact: {path}") from error
    if (
        stat.S_ISLNK(result.st_mode)
        or not stat.S_ISREG(result.st_mode)
        or bool(getattr(result, "st_reparse_tag", 0))
    ):
        raise RuntimeError(f"calibration cache artifact is not a regular file: {path}")
    if expected_size is not None and result.st_size != expected_size:
        raise RuntimeError(f"calibration cache artifact size mismatch: {path.name}")


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while publishing calibration cache metadata")
        offset += written


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _preallocate(path: Path, byte_size: int) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.ftruncate(descriptor, byte_size)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path, subject: str) -> dict[str, Any]:
    _validate_regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"calibration cache {subject} is invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"calibration cache {subject} is invalid")
    return value


def _journal_payload(
    layout_fingerprint: str, committed_rows: int, commits: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": _JOURNAL_SCHEMA,
        "layout_fingerprint": layout_fingerprint,
        "committed_rows": committed_rows,
        "commits": [dict(item) for item in commits],
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def _validate_journal(
    journal: Mapping[str, Any], layout: CacheLayout
) -> tuple[int, list[dict[str, Any]]]:
    value = dict(journal)
    fingerprint = value.pop("fingerprint", None)
    if (
        set(value)
        != {"schema_version", "layout_fingerprint", "committed_rows", "commits"}
        or value.get("schema_version") != _JOURNAL_SCHEMA
        or value.get("layout_fingerprint") != layout.fingerprint
        or fingerprint != _fingerprint(value)
    ):
        raise RuntimeError("calibration cache journal fingerprint mismatch")
    committed = value["committed_rows"]
    commits = value["commits"]
    if (
        isinstance(committed, bool)
        or not isinstance(committed, int)
        or committed < 0
        or committed > layout.row_count
        or not isinstance(commits, list)
    ):
        raise RuntimeError("calibration cache journal schema mismatch")
    expected_start = 0
    normalized: list[dict[str, Any]] = []
    for item in commits:
        if not isinstance(item, dict) or set(item) != {"start", "end", "sha256"}:
            raise RuntimeError("calibration cache journal range schema mismatch")
        start, end, digest = item["start"], item["end"], item["sha256"]
        if (
            type(start) is not int
            or type(end) is not int
            or start != expected_start
            or end <= start
            or end > committed
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("calibration cache journal range mismatch")
        expected_start = end
        normalized.append(dict(item))
    if expected_start != committed:
        raise RuntimeError("calibration cache journal committed prefix mismatch")
    return committed, normalized


def _range_digest(
    root: Path, specs: Mapping[str, _ArraySpec], start: int, end: int
) -> str:
    digest = hashlib.sha256()
    for name in sorted(specs):
        spec = specs[name]
        row_bytes = spec.byte_size // spec.shape[0]
        digest.update(name.encode("ascii"))
        digest.update(b"\x00")
        with (root / f"{name}.bin").open("rb", buffering=0) as handle:
            handle.seek(start * row_bytes)
            remaining = (end - start) * row_bytes
            while remaining:
                block = handle.read(min(1 << 20, remaining))
                if not block:
                    raise RuntimeError("calibration cache range is truncated")
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


def _verify_committed_ranges(
    root: Path, specs: Mapping[str, _ArraySpec], commits: Sequence[Mapping[str, Any]]
) -> None:
    for item in commits:
        if _range_digest(root, specs, int(item["start"]), int(item["end"])) != item[
            "sha256"
        ]:
            raise RuntimeError("calibration cache committed range fingerprint mismatch")


def _close_memmaps(arrays: Sequence[np.memmap[Any, Any]] | Any) -> None:
    for array in tuple(arrays):
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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


def _fsync_file(path: Path) -> None:
    _validate_regular_file(path)
    # Windows rejects FlushFileBuffers on a read-only descriptor.
    descriptor = os.open(path, os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_journal(root: Path, payload: Mapping[str, Any]) -> None:
    temporary = root / f".{_JOURNAL_FILE}.{uuid.uuid4().hex}.tmp"
    try:
        _write_exclusive(temporary, _json_bytes(payload))
        os.replace(temporary, root / _JOURNAL_FILE)
        _fsync_directory(root)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "CACHE_ALGORITHM",
    "CacheLayout",
    "CalibrationCache",
    "class_weights_from_counts",
]
