"""Bounded, deterministic CSV schema and row-normalization inspection."""

from __future__ import annotations

import csv
import datetime as dt
import itertools
import math
import os
import sqlite3
import stat
import tempfile
import threading
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from bitguard_bnn.constants import botiot_behavior, nbaiot_behavior, normalize_token


class SchemaInspectionError(RuntimeError):
    """A source schema or row cannot satisfy the full-data normalization contract."""


@dataclass(frozen=True, slots=True)
class RejectedRowSample:
    relative_path: str
    row_number: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "row_number": self.row_number,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class FileSchemaReport:
    relative_path: str
    rows: int
    accepted_rows: int
    rejected_rows: int
    columns: tuple[str, ...]
    class_counts: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "rows": self.rows,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "columns": list(self.columns),
            "class_counts": dict(self.class_counts),
        }


@dataclass(frozen=True, slots=True)
class SchemaInspectionReport:
    dataset: str
    root: str
    files: tuple[FileSchemaReport, ...]
    feature_columns: tuple[str, ...]
    total_rows: int
    accepted_rows: int
    rejected_rows: int
    class_counts: tuple[tuple[str, int], ...]
    unique_devices: int
    device_samples: tuple[tuple[str, int], ...]
    rejected_reasons: tuple[tuple[str, int], ...]
    rejected_samples: tuple[RejectedRowSample, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "root": self.root,
            "files": [item.as_dict() for item in self.files],
            "feature_columns": list(self.feature_columns),
            "total_rows": self.total_rows,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "class_counts": dict(self.class_counts),
            "unique_devices": self.unique_devices,
            "device_samples": dict(self.device_samples),
            "rejected_reasons": dict(self.rejected_reasons),
            "rejected_samples": [item.as_dict() for item in self.rejected_samples],
        }


_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_MISSING_NUMERIC = {"", "na", "n/a", "nan", "null", "none", "?"}
_DEFAULT_REQUIRED = {
    "nbaiot": (),
    "botiot": ("category", "subcategory", "saddr", "stime"),
}
_CSV_FIELD_LIMIT_LOCK = threading.RLock()


def _is_reparse(result: os.stat_result) -> bool:
    return bool((getattr(result, "st_file_attributes", 0) or 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _content_fingerprint(result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_size,
        getattr(result, "st_mtime_ns", int(result.st_mtime * 1_000_000_000)),
    )


def _regular_file(path: Path) -> os.stat_result:
    try:
        result = path.lstat()
    except OSError as error:
        raise SchemaInspectionError(f"cannot inspect CSV source {path}: {error}") from error
    if not stat.S_ISREG(result.st_mode) or stat.S_ISLNK(result.st_mode) or _is_reparse(result):
        raise SchemaInspectionError(f"CSV source must be a regular non-link file: {path}")
    return result


def _verify_source_directories(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise SchemaInspectionError(f"CSV source escapes its inspection root: {path}") from error
    current = root
    try:
        root_result = current.lstat()
    except OSError as error:
        raise SchemaInspectionError(
            f"CSV source root changed during inspection: {current}"
        ) from error
    if (
        not stat.S_ISDIR(root_result.st_mode)
        or stat.S_ISLNK(root_result.st_mode)
        or _is_reparse(root_result)
    ):
        raise SchemaInspectionError(f"CSV source root changed during inspection: {current}")
    directories = relative.parts[:-1]
    for component in directories:
        current /= component
        try:
            result = current.lstat()
        except OSError as error:
            raise SchemaInspectionError(
                f"CSV source directory changed during inspection: {current}"
            ) from error
        if (
            not stat.S_ISDIR(result.st_mode)
            or stat.S_ISLNK(result.st_mode)
            or _is_reparse(result)
        ):
            raise SchemaInspectionError(
                f"CSV source directory changed during inspection: {current}"
            )


def _open_pinned_text(path: Path) -> tuple[TextIO, os.stat_result, os.stat_result]:
    before = _regular_file(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SchemaInspectionError(f"cannot open CSV source {path}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_object(before, opened):
            raise SchemaInspectionError(f"CSV source changed during inspection: {path}")
        return os.fdopen(descriptor, "r", encoding="utf-8-sig", newline=""), opened, before
    except BaseException:
        os.close(descriptor)
        raise


def _verify_pinned_text(
    path: Path,
    handle: TextIO,
    opened: os.stat_result,
    path_before: os.stat_result,
    root: Path,
) -> None:
    _verify_source_directories(root, path)
    current_fd = os.fstat(handle.fileno())
    try:
        current_path = path.lstat()
    except OSError as error:
        raise SchemaInspectionError(f"CSV source changed during inspection: {path}") from error
    if (
        _content_fingerprint(opened) != _content_fingerprint(current_fd)
        or _content_fingerprint(path_before) != _content_fingerprint(current_path)
    ):
        raise SchemaInspectionError(f"CSV source changed during inspection: {path}")


def _discover_csvs(source: Path) -> tuple[Path, tuple[Path, ...]]:
    absolute = source.expanduser().absolute()
    try:
        root_result = absolute.lstat()
    except OSError as error:
        raise SchemaInspectionError(
            f"cannot inspect CSV source root {absolute}: {error}"
        ) from error
    if stat.S_ISREG(root_result.st_mode):
        _regular_file(absolute)
        if absolute.suffix.casefold() != ".csv":
            raise SchemaInspectionError(f"CSV source file must end in .csv: {absolute}")
        return absolute.parent, (absolute,)
    if (
        not stat.S_ISDIR(root_result.st_mode)
        or stat.S_ISLNK(root_result.st_mode)
        or _is_reparse(root_result)
    ):
        raise SchemaInspectionError(
            f"CSV source root must be a regular non-link directory: {absolute}"
        )

    files: list[Path] = []
    for base, directories, names in os.walk(absolute, topdown=True, followlinks=False):
        base_path = Path(base)
        for name in directories:
            directory = base_path / name
            result = directory.lstat()
            if (
                stat.S_ISLNK(result.st_mode)
                or _is_reparse(result)
                or not stat.S_ISDIR(result.st_mode)
            ):
                raise SchemaInspectionError(
                    f"CSV source tree contains a link, reparse point, or non-directory: {directory}"
                )
        for name in names:
            path = base_path / name
            if path.suffix.casefold() == ".csv":
                _regular_file(path)
                files.append(path)
    files.sort(key=lambda item: item.relative_to(absolute).as_posix().casefold())
    if not files:
        raise SchemaInspectionError(f"no CSV files found under {absolute}")
    relative_keys = [item.relative_to(absolute).as_posix().casefold() for item in files]
    if len(relative_keys) != len(set(relative_keys)):
        raise SchemaInspectionError("CSV source contains duplicate case-folded relative paths")
    return absolute, tuple(files)


class _BoundedLines:
    def __init__(self, handle: TextIO, max_record_chars: int) -> None:
        self.handle = handle
        self.max_record_chars = max_record_chars
        self.current_record_chars = 0

    def __iter__(self) -> _BoundedLines:
        return self

    def __next__(self) -> str:
        remaining = self.max_record_chars - self.current_record_chars
        line = self.handle.readline(remaining + 1)
        if not line:
            raise StopIteration
        self.current_record_chars += len(line)
        if self.current_record_chars > self.max_record_chars:
            raise SchemaInspectionError(
                f"CSV record exceeds max_record_chars={self.max_record_chars}"
            )
        return line

    def reset_record(self) -> None:
        self.current_record_chars = 0


class _DeviceStore:
    """Exact device counts on disk; report materialization remains explicitly capped."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE devices (device TEXT PRIMARY KEY, rows INTEGER NOT NULL) WITHOUT ROWID"
        )

    def add(self, counts: Counter[str]) -> None:
        self.connection.executemany(
            "INSERT INTO devices(device, rows) VALUES (?, ?) "
            "ON CONFLICT(device) DO UPDATE SET rows = rows + excluded.rows",
            sorted(counts.items()),
        )

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM devices").fetchone()
        assert row is not None
        return int(row[0])

    def samples(self, limit: int) -> tuple[tuple[str, int], ...]:
        return tuple(
            (str(device), int(rows))
            for device, rows in self.connection.execute(
                "SELECT device, rows FROM devices ORDER BY device LIMIT ?", (limit,)
            )
        )

    def close(self) -> None:
        self.connection.close()


def _header_mapping(header: Sequence[str], path: Path) -> dict[str, str]:
    if not header or all(not str(item).strip() for item in header):
        raise SchemaInspectionError(f"empty CSV header: {path}")
    if any(not str(item).strip() for item in header):
        raise SchemaInspectionError(f"CSV header contains an empty column: {path}")
    mapping: dict[str, str] = {}
    for column in header:
        if column != column.strip() or any(ord(character) < 32 for character in column):
            raise SchemaInspectionError(
                f"CSV header contains whitespace or control characters in {path}: {column!r}"
            )
        key = unicodedata.normalize("NFC", column).casefold()
        if key in mapping:
            raise SchemaInspectionError(
                f"duplicate column after case folding in {path}: {column!r}"
            )
        mapping[key] = column
    return mapping


def _column_key(column: str) -> str:
    return unicodedata.normalize("NFC", column).casefold()


def _find(mapping: dict[str, str], preferred: str | None, candidates: Iterable[str]) -> str | None:
    for name in itertools.chain((preferred,), candidates):
        key = _column_key(name) if name else None
        if key and key in mapping:
            return mapping[key]
    return None


def _numeric_reason(value: str) -> str | None:
    token = value.strip().casefold()
    if token in _MISSING_NUMERIC:
        return None
    try:
        number = float(token)
    except ValueError:
        return "non_numeric_feature"
    if not math.isfinite(number):
        return "non_finite_feature"
    return None


def _timestamp_valid(value: str) -> bool:
    token = value.strip()
    if not token:
        return False
    try:
        numeric = float(token)
    except ValueError:
        try:
            dt.datetime.fromisoformat(token.replace("Z", "+00:00"))
        except ValueError:
            return False
        return True
    return math.isfinite(numeric)


@dataclass(frozen=True, slots=True)
class _Schema:
    header: tuple[str, ...]
    features: tuple[str, ...]
    label: str | None
    raw_label: str | None
    device: str | None
    timestamp: str | None


def _schema_for(
    dataset: str,
    header: Sequence[str],
    required: Sequence[str],
    path: Path,
) -> _Schema:
    mapping = _header_mapping(header, path)
    missing = [
        column
        for column in required
        if _column_key(column) not in mapping
    ]
    if missing:
        raise SchemaInspectionError(
            f"missing required columns in {path}: {sorted(missing, key=str.casefold)}"
        )
    if dataset == "nbaiot":
        features = tuple(sorted(header, key=str.casefold))
        return _Schema(tuple(header), features, None, None, None, None)

    label = _find(mapping, None, ("category", "label", "attack"))
    raw_label = _find(mapping, None, ("subcategory", "attack", "category", "label"))
    device = _find(mapping, None, ("saddr", "srcip", "device_id"))
    timestamp = _find(mapping, None, ("stime", "timestamp", "time"))
    metadata = {item for item in (label, raw_label, device, timestamp) if item is not None}
    features = tuple(
        sorted((column for column in header if column not in metadata), key=str.casefold)
    )
    if not features:
        raise SchemaInspectionError(f"no numeric feature columns remain in {path}")
    return _Schema(tuple(header), features, label, raw_label, device, timestamp)


def _nbaiot_metadata(root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    device = relative.parts[0] if len(relative.parts) > 1 else path.parent.name
    stem = normalize_token(path.stem)
    if "benign" in stem:
        raw_attack = "benign"
    else:
        family = normalize_token(path.parent.name.replace("_attacks", ""))
        raw_attack = f"{family}_{stem}"
    return str(device), nbaiot_behavior(raw_attack)


def _row_metadata(
    dataset: str,
    values: dict[str, str],
    schema: _Schema,
    root: Path,
    path: Path,
) -> tuple[str, str] | str:
    if dataset == "nbaiot":
        return _nbaiot_metadata(root, path)
    if schema.label is None:
        return "invalid_label"
    category = values[schema.label].strip()
    raw = values[schema.raw_label].strip() if schema.raw_label else category
    if not category:
        return "invalid_label"
    label = botiot_behavior(category, raw)
    if schema.device is None:
        device = f"source_{path.stem}"
    else:
        device = values[schema.device].strip()
        if not device:
            return "invalid_device"
    if schema.timestamp is not None and not _timestamp_valid(values[schema.timestamp]):
        return "invalid_timestamp"
    return device, label


def _row_reason(
    row: Sequence[str],
    schema: _Schema,
    *,
    max_record_chars: int,
) -> tuple[str | None, dict[str, str] | None]:
    if len(row) != len(schema.header):
        return "column_count_mismatch", None
    if sum(len(item) for item in row) > max_record_chars:
        raise SchemaInspectionError(
            f"CSV logical record exceeds max_record_chars={max_record_chars}"
        )
    values = dict(zip(schema.header, row))
    for feature in schema.features:
        reason = _numeric_reason(values[feature])
        if reason is not None:
            return reason, values
    return None, values


def _next_chunk(
    reader: Iterable[list[str]],
    lines: _BoundedLines,
    chunk_size: int,
) -> list[list[str]]:
    iterator = iter(reader)
    chunk: list[list[str]] = []
    for _ in range(chunk_size):
        try:
            row = next(iterator)
        except StopIteration:
            break
        lines.reset_record()
        chunk.append(row)
    return chunk


def _inspect_csv_dataset_unlocked(
    dataset: str,
    source: str | os.PathLike[str],
    *,
    required_columns: Sequence[str] | None = None,
    chunk_size: int = 10_000,
    max_record_chars: int = 1 << 20,
    fail_on_rejected: bool = True,
    rejected_sample_limit: int = 32,
    device_sample_limit: int = 128,
) -> SchemaInspectionReport:
    """Inspect all CSV rows in bounded chunks without retaining a complete frame."""

    normalized_dataset = str(dataset).strip().casefold()
    if normalized_dataset not in _DEFAULT_REQUIRED:
        raise ValueError(f"unsupported dataset for schema inspection: {dataset!r}")
    for name, value in (
        ("chunk_size", chunk_size),
        ("max_record_chars", max_record_chars),
        ("rejected_sample_limit", rejected_sample_limit),
        ("device_sample_limit", device_sample_limit),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(required_columns, (str, bytes)):
        raise ValueError("required_columns must be a sequence of column names")
    if not isinstance(fail_on_rejected, bool):
        raise ValueError("fail_on_rejected must be a boolean")
    required = (
        tuple(required_columns)
        if required_columns is not None
        else _DEFAULT_REQUIRED[normalized_dataset]
    )
    if any(not isinstance(item, str) or not item.strip() for item in required):
        raise ValueError("required_columns must contain only non-empty strings")
    if len({_column_key(item) for item in required}) != len(required):
        raise ValueError("required_columns must not contain case-folded duplicates")
    root, paths = _discover_csvs(Path(source))
    canonical_features: tuple[str, ...] | None = None
    file_reports: list[FileSchemaReport] = []
    total_rows = 0
    accepted_rows = 0
    rejection_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    finite_feature_counts: Counter[str] = Counter()
    samples: list[RejectedRowSample] = []

    with tempfile.TemporaryDirectory(prefix="bitguard-schema-") as temp_directory:
        devices = _DeviceStore(Path(temp_directory) / "devices.sqlite3")
        try:
            for path in paths:
                relative = path.relative_to(root).as_posix()
                _verify_source_directories(root, path)
                handle, opened, path_before = _open_pinned_text(path)
                file_rows = 0
                file_accepted = 0
                file_rejections: Counter[str] = Counter()
                file_classes: Counter[str] = Counter()
                try:
                    bounded_lines = _BoundedLines(handle, max_record_chars)
                    reader = csv.reader(bounded_lines, strict=True)
                    try:
                        header = next(reader)
                    except StopIteration as error:
                        raise SchemaInspectionError(f"empty CSV: {path}") from error
                    bounded_lines.reset_record()
                    schema = _schema_for(normalized_dataset, header, required, path)
                    if canonical_features is None:
                        canonical_features = schema.features
                    elif {_column_key(item) for item in schema.features} != {
                        _column_key(item) for item in canonical_features
                    }:
                        raise SchemaInspectionError(
                            f"feature schema mismatch in {path}: "
                            f"expected={list(canonical_features)}, observed={list(schema.features)}"
                        )

                    logical_row = 1
                    while True:
                        chunk = _next_chunk(reader, bounded_lines, chunk_size)
                        if not chunk:
                            break
                        chunk_devices: Counter[str] = Counter()
                        for row in chunk:
                            logical_row += 1
                            file_rows += 1
                            total_rows += 1
                            reason, values = _row_reason(
                                row, schema, max_record_chars=max_record_chars
                            )
                            if reason is None:
                                assert values is not None
                                metadata = _row_metadata(
                                    normalized_dataset, values, schema, root, path
                                )
                                if isinstance(metadata, str):
                                    reason = metadata
                                else:
                                    device, label = metadata
                                    file_accepted += 1
                                    accepted_rows += 1
                                    file_classes[label] += 1
                                    class_counts[label] += 1
                                    chunk_devices[device] += 1
                                    for feature in schema.features:
                                        token = values[feature].strip().casefold()
                                        if token not in _MISSING_NUMERIC:
                                            finite_feature_counts[_column_key(feature)] += 1
                            if reason is not None:
                                file_rejections[reason] += 1
                                rejection_counts[reason] += 1
                                if len(samples) < rejected_sample_limit:
                                    samples.append(
                                        RejectedRowSample(relative, logical_row, reason)
                                    )
                        devices.add(chunk_devices)
                except (csv.Error, UnicodeError) as error:
                    raise SchemaInspectionError(f"cannot parse CSV {path}: {error}") from error
                finally:
                    try:
                        _verify_pinned_text(path, handle, opened, path_before, root)
                    finally:
                        handle.close()
                file_reports.append(
                    FileSchemaReport(
                        relative_path=relative,
                        rows=file_rows,
                        accepted_rows=file_accepted,
                        rejected_rows=sum(file_rejections.values()),
                        columns=tuple(header),
                        class_counts=tuple(sorted(file_classes.items())),
                    )
                )

            if total_rows == 0:
                raise SchemaInspectionError("CSV dataset contains headers but no data rows")
            rejected_rows = sum(rejection_counts.values())
            if fail_on_rejected and rejected_rows:
                reasons = ", ".join(
                    f"{reason}={count}" for reason, count in sorted(rejection_counts.items())
                )
                raise SchemaInspectionError(
                    f"schema inspection found {rejected_rows} rejected rows: {reasons}"
                )
            assert canonical_features is not None
            unusable = [
                feature
                for feature in canonical_features
                if not finite_feature_counts[_column_key(feature)]
            ]
            if accepted_rows and unusable:
                raise SchemaInspectionError(
                    "numeric feature columns contain no finite values: "
                    f"{sorted(unusable, key=str.casefold)}"
                )
            return SchemaInspectionReport(
                dataset=normalized_dataset,
                root=str(root),
                files=tuple(file_reports),
                feature_columns=canonical_features,
                total_rows=total_rows,
                accepted_rows=accepted_rows,
                rejected_rows=rejected_rows,
                class_counts=tuple(sorted(class_counts.items())),
                unique_devices=devices.count(),
                device_samples=devices.samples(device_sample_limit),
                rejected_reasons=tuple(sorted(rejection_counts.items())),
                rejected_samples=tuple(samples),
            )
        finally:
            devices.close()


def inspect_csv_dataset(
    dataset: str,
    source: str | os.PathLike[str],
    *,
    required_columns: Sequence[str] | None = None,
    chunk_size: int = 10_000,
    max_record_chars: int = 1 << 20,
    fail_on_rejected: bool = True,
    rejected_sample_limit: int = 32,
    device_sample_limit: int = 128,
) -> SchemaInspectionReport:
    """Inspect all CSV rows while isolating the process-global CSV field limit."""

    if (
        isinstance(max_record_chars, bool)
        or not isinstance(max_record_chars, int)
        or max_record_chars <= 0
    ):
        raise ValueError("max_record_chars must be a positive integer")
    with _CSV_FIELD_LIMIT_LOCK:
        previous_limit = csv.field_size_limit()
        if previous_limit < max_record_chars:
            csv.field_size_limit(max_record_chars)
        try:
            return _inspect_csv_dataset_unlocked(
                dataset,
                source,
                required_columns=required_columns,
                chunk_size=chunk_size,
                max_record_chars=max_record_chars,
                fail_on_rejected=fail_on_rejected,
                rejected_sample_limit=rejected_sample_limit,
                device_sample_limit=device_sample_limit,
            )
        finally:
            csv.field_size_limit(previous_limit)
