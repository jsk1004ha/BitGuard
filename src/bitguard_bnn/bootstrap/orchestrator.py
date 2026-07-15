"""Idempotent acquisition orchestration through verified CSV sources."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from .cleanup import inspection_command, scan_cleanup_debt
from .download import DownloadResult, download_file
from .extract import ExtractionResult, extract_rar, extract_zip
from .fsops import rename_directory_noreplace
from .inspect import SchemaInspectionReport, inspect_csv_dataset
from .manifest import build_source_manifest, write_source_manifest
from .preflight import (
    ArchiveInspection,
    choose_compute,
    discover_cpu,
    discover_ram,
    estimate_resources,
    probe_nvidia_driver,
    require_disk,
    verify_torch_compute,
)
from .registry import load_registry
from .state import (
    STAGE_ORDER,
    BootstrapStateStore,
    BootstrapWriterLock,
    _fsync_parent_directory,
)
from .types import BootstrapOptions, DatasetSpec


REPORT_FORMAT_VERSION = 1
JOURNAL_FORMAT_VERSION = 1
UNKNOWN_REMOTE_ARCHIVE_BYTES = 4 * 1024**3
ARCHIVE_EXPANSION_FACTOR = 12
REPORT_AND_METADATA_BYTES = 64 * 1024**2
DEFAULT_DISK_RESERVE_BYTES = 2 * 1024**3
_URLISH_PATH = re.compile(
    r"(?<![A-Za-z0-9+.-])(?![A-Za-z]:[\\/])"
    r"[A-Za-z][A-Za-z0-9+.-]*:(?:[\\/]+|[^\s]*[@?#][^\s]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    input_signature: Callable[[], str]
    run: Callable[[], Sequence[Path]]
    always_run: bool = False


class SourceContext(TypedDict):
    kind: str
    digest: str | None
    bytes: int
    source: Path | None


def _group_preparation_disk_requirements(
    estimates: Mapping[str, object],
    *,
    work_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    device_for: Callable[[Path], int],
) -> dict[int, dict[str, object]]:
    """Group preparation bytes where their temporary and final files live."""

    groups: dict[int, dict[str, object]] = {}

    def add(path: Path, required: int, dataset: str, kind: str) -> None:
        device = int(device_for(path))
        group = groups.setdefault(
            device,
            {"path": str(path), "required_bytes": 0, "datasets": {}},
        )
        group["required_bytes"] = cast(int, group["required_bytes"]) + required
        datasets = cast(dict[str, dict[str, int]], group["datasets"])
        values = datasets.setdefault(dataset, {})
        values[kind] = values.get(kind, 0) + required

    for dataset, estimate in estimates.items():
        work_bytes = sum(
            int(getattr(estimate, name))
            for name in (
                "source_snapshot_bytes",
                "membership_sqlite_bytes",
                "audit_sqlite_bytes",
            )
        )
        output_bytes = sum(
            int(getattr(estimate, name))
            for name in (
                "external_merge_bytes",
                "staging_bytes",
                "final_shard_bytes",
            )
        )
        add(work_paths[dataset], work_bytes, dataset, "work")
        add(output_paths[dataset], output_bytes, dataset, "output")
    return groups


def _default_compute_resolver(requested: str) -> dict[str, object]:
    driver = probe_nvidia_driver()
    if requested == "cpu":
        selected = "cpu"
    else:
        try:
            import torch
        except Exception as error:  # pragma: no cover - installation failure path
            raise RuntimeError(
                f"Torch import failed during compute selection: {error}"
            ) from error
        selected = choose_compute(
            requested,
            driver=driver,
            torch_cuda=bool(torch.cuda.is_available()),
            torch_cuda_version=torch.version.cuda,
        )
    verification = verify_torch_compute(selected, driver=driver)
    return {
        "requested": requested,
        "selected_profile": selected,
        "driver": driver.as_dict(),
        "verification": verification.as_dict(),
        "cpu": discover_cpu().as_dict(),
        "ram": discover_ram().as_dict(),
    }


@dataclass(frozen=True, slots=True)
class BootstrapDependencies:
    """Narrow test injection surface; production defaults retain official sources."""

    nbaiot_archive: Path | None = None
    nbaiot_download_size: int | None = None
    available_bytes: int | None = None
    preparation_available_bytes: int | None = None
    downloader: Callable[..., DownloadResult] = download_file
    zip_extractor: Callable[..., ExtractionResult] = extract_zip
    rar_extractor: Callable[..., ExtractionResult] = extract_rar
    inspector: Callable[..., SchemaInspectionReport] = inspect_csv_dataset
    compute_resolver: Callable[[str], Mapping[str, object]] = _default_compute_resolver
    preparer: Callable[..., object] | None = None
    prepared_verifier: Callable[[Path], object] | None = None
    preparation_signature_token: str | None = None


def _json_signature(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_digest(path: Path) -> tuple[str, int]:
    first = path.lstat()
    if stat.S_ISLNK(first.st_mode) or not stat.S_ISREG(first.st_mode):
        raise RuntimeError(
            f"Bootstrap source must be a regular non-symlink file: {path}"
        )
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        opened = os.fstat(stream.fileno())
    final = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(first) != identity(opened) or identity(first) != identity(final):
        raise RuntimeError(f"Bootstrap source changed while it was hashed: {path}")
    return digest.hexdigest(), size


def _tree_digest(path: Path) -> tuple[str, int, tuple[Path, ...]]:
    supplied = path.lstat()
    if stat.S_ISLNK(supplied.st_mode) or not stat.S_ISDIR(supplied.st_mode):
        raise RuntimeError(f"Bootstrap source must be a non-symlink directory: {path}")
    root = path.resolve(strict=True)
    resolved = root.lstat()
    supplied_identity = (supplied.st_dev, supplied.st_ino, supplied.st_mode)
    resolved_identity = (resolved.st_dev, resolved.st_ino, resolved.st_mode)
    if supplied_identity != resolved_identity:
        raise RuntimeError(f"Bootstrap source changed while it was resolved: {path}")
    records: list[tuple[str, int, str]] = []
    files: list[Path] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        result = candidate.lstat()
        if stat.S_ISLNK(result.st_mode):
            raise RuntimeError(f"Bootstrap source links are not allowed: {candidate}")
        if stat.S_ISDIR(result.st_mode):
            continue
        if not stat.S_ISREG(result.st_mode):
            raise RuntimeError(f"Unsupported bootstrap source entry: {candidate}")
        digest, size = _regular_digest(candidate)
        records.append((candidate.relative_to(root).as_posix(), size, digest))
        files.append(candidate)
    if not records:
        raise RuntimeError(
            f"Bootstrap source directory contains no regular files: {root}"
        )
    return _json_signature(records), sum(item[1] for item in records), tuple(files)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("atomic JSON write made no progress")
        offset += written


class AtomicJsonWriteError(RuntimeError):
    """Describe whether an atomic JSON failure happened before or after replace."""

    def __init__(
        self,
        path: Path,
        *,
        published: bool,
        published_identity: tuple[int, int] | None,
        cause: BaseException,
    ) -> None:
        phase = "after publication" if published else "before publication"
        super().__init__(f"Atomic JSON write for {path} failed {phase}: {cause}")
        self.path = path
        self.published = published
        self.published_identity = published_identity
        self.cause = cause


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    published = False
    published_identity: tuple[int, int] | None = None
    try:
        _ensure_durable_directory(path.parent)
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        operation_error: BaseException | None = None
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise RuntimeError(
                    f"Atomic JSON temporary is not a regular file: {temporary}"
                )
            published_identity = (int(opened.st_dev), int(opened.st_ino))
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        except BaseException as error:
            operation_error = error
        try:
            os.close(descriptor)
        except OSError as close_error:
            if operation_error is None:
                operation_error = close_error
            else:
                operation_error = OSError(
                    f"{operation_error}; temporary close also failed: {close_error}"
                )
        if operation_error is not None:
            raise operation_error
        temporary.replace(path)
        published = True
        committed = path.lstat()
        if (
            not stat.S_ISREG(committed.st_mode)
            or stat.S_ISLNK(committed.st_mode)
            or (int(committed.st_dev), int(committed.st_ino)) != published_identity
        ):
            raise RuntimeError(
                f"Atomic JSON publication identity could not be verified: {path}"
            )
        _fsync_parent_directory(path)
    except AtomicJsonWriteError:
        raise
    except BaseException as error:
        raise AtomicJsonWriteError(
            path,
            published=published,
            published_identity=published_identity if published else None,
            cause=error,
        ) from error


def _invalidate_published_json(
    path: Path,
    expected_identity: tuple[int, int] | None,
) -> None:
    """Remove a writer-owned success envelope when atomic reconciliation cannot publish."""

    if expected_identity is None:
        raise RuntimeError(
            f"Cannot invalidate published JSON {path} without a verified identity"
        )
    descriptor = os.open(
        path,
        os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
    )
    operation_error: BaseException | None = None
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or (int(current.st_dev), int(current.st_ino)) != expected_identity
        ):
            raise RuntimeError(
                f"Published JSON identity changed before invalidation: {path}"
            )
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
    except BaseException as error:
        operation_error = error
    try:
        os.close(descriptor)
    except OSError as close_error:
        if operation_error is None:
            operation_error = close_error
        else:
            operation_error = OSError(
                f"{operation_error}; invalidated JSON close also failed: {close_error}"
            )
    if operation_error is not None:
        raise operation_error


def _fsync_directory(path: Path) -> None:
    """Durably flush a directory where the platform exposes directory fds."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    sync_error: OSError | None = None
    try:
        os.fsync(descriptor)
    except OSError as error:
        sync_error = error
    try:
        os.close(descriptor)
    except OSError as close_error:
        if sync_error is not None:
            raise OSError(
                f"directory fsync failed: {sync_error}; directory close failed: "
                f"{close_error}"
            ) from sync_error
        raise
    if sync_error is not None:
        raise sync_error


def _ensure_durable_directory(path: Path) -> bool:
    """Create one protocol root and durably publish it before it is used."""

    def unsafe(result: os.stat_result) -> bool:
        return (
            stat.S_ISLNK(result.st_mode)
            or not stat.S_ISDIR(result.st_mode)
            or bool(getattr(result, "st_reparse_tag", 0))
        )

    try:
        existing = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise RuntimeError(
            f"Cannot inspect protocol directory {path}: {error}"
        ) from error
    else:
        if unsafe(existing):
            raise RuntimeError(
                f"Protocol directory must be a non-symlink directory: {path}"
            )
        _fsync_directory(path)
        _fsync_parent_directory(path)
        return False

    parent = path.parent
    if parent == path:
        raise RuntimeError(f"Cannot create missing protocol filesystem root: {path}")
    try:
        parent_result = parent.lstat()
    except FileNotFoundError:
        _ensure_durable_directory(parent)
        parent_result = parent.lstat()
    except OSError as error:
        raise RuntimeError(
            f"Cannot inspect protocol directory parent {parent}: {error}"
        ) from error
    if unsafe(parent_result):
        raise RuntimeError(
            f"Protocol directory parent must be a non-symlink directory: {parent}"
        )
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError:
        current = path.lstat()
        if unsafe(current):
            raise RuntimeError(
                f"Protocol directory must be a non-symlink directory: {path}"
            )
        _fsync_directory(path)
        _fsync_parent_directory(path)
        return False
    created = path.lstat()
    if unsafe(created):
        raise RuntimeError(f"Created protocol directory changed type: {path}")
    _fsync_directory(path)
    _fsync_parent_directory(path)
    return True


def _fsync_regular_file(path: Path) -> None:
    """Flush one unchanged regular file and reject link/reparse substitution."""

    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or bool(getattr(before, "st_reparse_tag", 0))
    ):
        raise RuntimeError(f"Durability target must be a regular file: {path}")
    object_identity = lambda result: (
        result.st_dev,
        result.st_ino,
        stat.S_IFMT(result.st_mode),
        result.st_size,
        result.st_mtime_ns,
    )
    original_mode = stat.S_IMODE(before.st_mode)
    mode_changed = False
    descriptor = -1
    operation_error: BaseException | None = None
    opened: os.stat_result | None = None
    try:
        if os.name == "nt" and not original_mode & stat.S_IWRITE:
            os.chmod(path, original_mode | stat.S_IWRITE)
            mode_changed = True
            writable = path.lstat()
            if object_identity(writable) != object_identity(before):
                raise RuntimeError(
                    f"Read-only durability target changed while enabling sync: {path}"
                )
        flags = (os.O_RDWR if os.name == "nt" else os.O_RDONLY) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if object_identity(opened) != object_identity(before):
            raise RuntimeError(
                f"Durability target changed before it was flushed: {path}"
            )
        os.fsync(descriptor)
    except BaseException as error:
        operation_error = error
    if descriptor >= 0:
        try:
            os.close(descriptor)
        except OSError as close_error:
            if operation_error is None:
                operation_error = close_error
            else:
                operation_error = OSError(
                    f"file durability failed: {operation_error}; file close failed: "
                    f"{close_error}"
                )
    restore_error: BaseException | None = None
    if mode_changed:
        try:
            current = path.lstat()
            if object_identity(current) != object_identity(before):
                raise RuntimeError(
                    f"Read-only durability target changed before mode restore: {path}"
                )
            os.chmod(path, original_mode)
        except BaseException as error:
            restore_error = error
    if operation_error is not None:
        if restore_error is not None:
            raise RuntimeError(
                f"File durability failed: {operation_error}; restoring read-only mode "
                f"also failed: {restore_error}"
            ) from operation_error
        raise operation_error
    if restore_error is not None:
        raise restore_error
    after = path.lstat()
    if (
        opened is None
        or object_identity(before) != object_identity(opened)
        or object_identity(before) != object_identity(after)
        or stat.S_IMODE(after.st_mode) != original_mode
    ):
        raise RuntimeError(f"Durability target changed while it was flushed: {path}")


def _make_tree_durable(root: Path) -> None:
    """Flush regular files and directories bottom-up before tree publication."""

    def visit(directory: Path) -> None:
        result = directory.lstat()
        if (
            stat.S_ISLNK(result.st_mode)
            or not stat.S_ISDIR(result.st_mode)
            or bool(getattr(result, "st_reparse_tag", 0))
        ):
            raise RuntimeError(
                f"Durability tree must contain only non-link directories: {directory}"
            )
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise RuntimeError(
                f"Cannot scan durability tree {directory}: {error}"
            ) from error
        for entry in entries:
            candidate = Path(entry.path)
            child = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(child.st_mode) or bool(getattr(child, "st_reparse_tag", 0)):
                raise RuntimeError(
                    f"Durability tree links are not allowed: {candidate}"
                )
            if stat.S_ISDIR(child.st_mode):
                visit(candidate)
            elif stat.S_ISREG(child.st_mode):
                _fsync_regular_file(candidate)
            else:
                raise RuntimeError(
                    f"Durability tree contains an unsupported entry: {candidate}"
                )
        _fsync_directory(directory)

    visit(root)
    _fsync_parent_directory(root)


def _make_content_durable(path: Path, kind: str) -> None:
    if kind == "directory":
        _make_tree_durable(path)
        return
    _fsync_regular_file(path)
    _fsync_parent_directory(path)


def _redact_text(value: str) -> str:
    match = _URLISH_PATH.search(value)
    if match is None:
        return value
    return value[: match.start()] + "<redacted-url>"


def _redact_mapping_key(value: object) -> object:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    return _redact_text(value) if isinstance(value, str) else value


def _redact_report_value(value: object) -> object:
    """Recursively remove URL credentials and URL query/fragment material."""

    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Mapping):
        redacted: dict[object, object] = {}
        for key, item in value.items():
            safe_key = _redact_mapping_key(key)
            if safe_key in redacted:
                suffix = 2
                candidate = f"{safe_key}#collision-{suffix}"
                while candidate in redacted:
                    suffix += 1
                    candidate = f"{safe_key}#collision-{suffix}"
                safe_key = candidate
            redacted[safe_key] = _redact_report_value(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact_report_value(item) for item in value]
    return value


def _safe_error(error: BaseException) -> str:
    return _redact_text(f"{type(error).__name__}: {error}")


def _sanitized_report_mapping(value: object, subject: str) -> dict[str, object]:
    sanitized = _redact_report_value(value)
    if not isinstance(sanitized, dict):
        raise RuntimeError(f"{subject} report payload must be a mapping")
    if not all(isinstance(key, str) for key in sanitized):
        raise RuntimeError(f"{subject} report payload keys must be strings")
    return cast(dict[str, object], sanitized)


def _environment_manifest(
    options: BootstrapOptions,
    resolved_compute: Mapping[str, object],
) -> dict[str, object]:
    """Return stable runtime facts without hostnames, users, or arbitrary fields."""

    verification_value = resolved_compute.get("verification")
    verification = verification_value if isinstance(verification_value, Mapping) else {}
    driver_value = resolved_compute.get("driver")
    driver = driver_value if isinstance(driver_value, Mapping) else {}
    cpu_value = resolved_compute.get("cpu")
    cpu = cpu_value if isinstance(cpu_value, Mapping) else {}
    ram_value = resolved_compute.get("ram")
    ram = ram_value if isinstance(ram_value, Mapping) else {}

    def fact(name: str) -> object:
        value = verification.get(name)
        return resolved_compute.get(name) if value is None else value

    torch_version = fact("torch_version")
    if not isinstance(torch_version, str) or not torch_version:
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None

    selected_profile = fact("selected_profile")
    manifest: dict[str, object] = {
        "runtime": {
            "os": {
                "system": platform.system(),
                "platform": sys.platform,
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "torch": {"version": torch_version},
            "cuda": {
                "runtime": fact("torch_cuda_version"),
                "device": fact("device"),
                "device_name": fact("device_name"),
                "device_index": fact("device_index"),
                "profile": selected_profile,
                "driver_version": driver.get("driver_version"),
                "driver_device_name": driver.get("device_name"),
                "driver_memory_bytes": driver.get("memory_bytes"),
                "driver_cuda_profile": driver.get("cuda_profile"),
            },
        },
        "compute": {
            "requested": resolved_compute.get("requested", options.compute),
            "selected_profile": selected_profile,
            "device": fact("device"),
            "device_name": fact("device_name"),
            "device_index": fact("device_index"),
            "cpu_logical_count": cpu.get("logical_count"),
            "ram_total_bytes": ram.get("total_bytes"),
        },
        "data_root_identity": _json_signature(
            {"resolved_data_root": str(options.data_root.resolve(strict=False))}
        ),
    }
    return manifest


def _resolved_local_path(path: Path, label: str) -> Path:
    raw = str(path)
    if _URLISH_PATH.search(raw):
        raise RuntimeError(
            f"{label} must be a local filesystem path; URL-looking values are not allowed"
        )
    return path.expanduser().resolve(strict=False)


def _existing_parent(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RuntimeError(
                f"No existing parent is available for bootstrap path {path}"
            )
        candidate = candidate.parent
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(
            f"Bootstrap parent must be an existing trusted directory: {candidate}"
        )
    return candidate.resolve(strict=True)


def _lock_path(options: BootstrapOptions) -> Path:
    data_root = _resolved_local_path(options.data_root, "data_root")
    parent = _existing_parent(data_root.parent)
    identity = _json_signature({"data_root": str(data_root)})[:20]
    return parent / f".bitguard-bootstrap-{identity}.lock"


def _load_journal(path: Path, subject: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        result = path.lstat()
        if stat.S_ISLNK(result.st_mode) or not stat.S_ISREG(result.st_mode):
            raise RuntimeError(
                f"{subject} journal must be a regular non-symlink file: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read {subject} journal {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{subject} journal must contain a JSON object: {path}")
    return payload


def _require_journal_shape(
    record: Mapping[str, object],
    *,
    subject: str,
    fields: set[str],
) -> None:
    if set(record) != fields:
        raise RuntimeError(f"{subject} journal has an invalid field set")
    if record["version"] != JOURNAL_FORMAT_VERSION:
        raise RuntimeError(f"{subject} journal uses an unsupported version")
    if record["status"] not in {"intent", "completed"}:
        raise RuntimeError(f"{subject} journal has an invalid status")
    for field in ("dataset", "final_path", "content_sha256"):
        value = record[field]
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{subject} journal has an invalid {field}")
    candidate_path = record["candidate_path"]
    if candidate_path is not None and (
        not isinstance(candidate_path, str) or not candidate_path
    ):
        raise RuntimeError(f"{subject} journal has an invalid candidate_path")
    if not isinstance(record["report"], dict):
        raise RuntimeError(f"{subject} journal has an invalid report payload")


def _content_fingerprint(path: Path, kind: str) -> tuple[str, int]:
    if kind == "directory":
        digest, size, _files = _tree_digest(path)
        return digest, size
    return _regular_digest(path)


def _published_outputs(path: Path, kind: str) -> tuple[Path, ...]:
    if kind == "directory":
        _digest, _size, files = _tree_digest(path)
        return files
    _regular_digest(path)
    return (path,)


def _retained_candidate(
    value: str,
    *,
    destination: Path,
    prefix: str,
) -> Path | None:
    supplied = Path(value)
    try:
        first = supplied.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(
            f"Cannot inspect retained journal candidate {supplied}: {error}"
        ) from error
    if stat.S_ISLNK(first.st_mode):
        raise RuntimeError(
            f"Retained journal candidate is outside the private staging namespace: {supplied}"
        )
    expected_parent = destination.parent.resolve(strict=True)
    try:
        supplied_parent = supplied.parent.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"Cannot resolve retained journal candidate parent {supplied.parent}: {error}"
        ) from error
    if (
        supplied_parent != expected_parent
        or re.fullmatch(
            rf"{re.escape(prefix)}[0-9a-f]{{32}}",
            supplied.name,
        )
        is None
    ):
        raise RuntimeError(
            f"Retained journal candidate is outside the private staging namespace: {supplied}"
        )
    candidate = expected_parent / supplied.name
    current = candidate.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_mode)
    if identity(first) != identity(current):
        raise RuntimeError(
            f"Retained journal candidate changed during validation: {candidate}"
        )
    return candidate


def _publish_candidate(candidate: Path, destination: Path, kind: str) -> None:
    _ensure_durable_directory(destination.parent)
    initial = candidate.lstat()
    if stat.S_ISLNK(initial.st_mode):
        raise RuntimeError(f"Publication candidate must not be a symlink: {candidate}")
    expected_mode = stat.S_IFDIR if kind == "directory" else stat.S_IFREG
    if stat.S_IFMT(initial.st_mode) != expected_mode:
        raise RuntimeError(f"Publication candidate has an invalid type: {candidate}")
    identity = lambda item: (item.st_dev, item.st_ino, stat.S_IFMT(item.st_mode))
    if kind == "directory":
        rename_directory_noreplace(candidate, destination)
        published = destination.lstat()
        if identity(published) != identity(initial):
            raise RuntimeError(
                f"Published directory has a foreign identity: {destination}"
            )
        _fsync_parent_directory(destination)
        return
    try:
        os.link(candidate, destination, follow_symlinks=False)
    except TypeError:  # pragma: no cover - legacy platform fallback
        os.link(candidate, destination)
    published = destination.lstat()
    if identity(published) != identity(initial):
        raise RuntimeError(f"Published file has a foreign identity: {destination}")
    _fsync_parent_directory(destination)


def _fallback_report_paths(options: BootstrapOptions) -> tuple[Path, ...]:
    raw_data_root = str(options.data_root)
    identity = _json_signature({"data_root": raw_data_root})[:20]
    cwd = Path.cwd().resolve(strict=True)
    parents: list[Path] = []
    if not _URLISH_PATH.search(raw_data_root):
        try:
            resolved_parent = (
                options.data_root.expanduser().resolve(strict=False).parent
            )
            parents.append(_existing_parent(resolved_parent))
        except BaseException:
            pass
    parents.append(cwd)
    unique: list[Path] = []
    for parent in parents:
        candidate = parent / f".bitguard-bootstrap-{identity}-failure-report.json"
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _copy_file(source: Path, destination: Path, expected_digest: str) -> Path:
    _ensure_durable_directory(destination.parent)
    if destination.exists():
        actual, _size = _regular_digest(destination)
        if actual == expected_digest:
            return destination
        raise RuntimeError(
            f"Existing acquisition output differs from its source: {destination}. "
            "Preserve it for inspection and use a new data root."
        )
    temporary = destination.with_name(f".bitguard-extract-copy-{uuid.uuid4().hex}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        copied_digest, _size = _regular_digest(temporary)
        if copied_digest != expected_digest:
            raise RuntimeError(f"Copied source failed digest verification: {source}")
        os.rename(temporary, destination)
        _fsync_parent_directory(destination)
    except BaseException:
        # The private candidate is intentionally retained if publication fails.
        raise
    return destination


def _copy_tree(
    source: Path, destination: Path, expected_digest: str
) -> tuple[Path, ...]:
    if destination.exists():
        actual, _size, files = _tree_digest(destination)
        if actual == expected_digest:
            return files
        raise RuntimeError(
            f"Existing acquisition directory differs from its source: {destination}. "
            "Preserve it for inspection and use a new data root."
        )
    _ensure_durable_directory(destination.parent)
    staging = Path(
        tempfile.mkdtemp(prefix=".bitguard-extract-copy-", dir=destination.parent)
    )
    source_root = source.resolve(strict=True)
    for candidate in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(source_root)
        target = staging / relative
        result = candidate.lstat()
        if stat.S_ISLNK(result.st_mode):
            raise RuntimeError(f"Bootstrap source links are not allowed: {candidate}")
        if stat.S_ISDIR(result.st_mode):
            target.mkdir(exist_ok=True)
        elif stat.S_ISREG(result.st_mode):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target, follow_symlinks=False)
        else:
            raise RuntimeError(f"Unsupported bootstrap source entry: {candidate}")
    copied_digest, _size, _files = _tree_digest(staging)
    if copied_digest != expected_digest:
        raise RuntimeError(f"Copied directory failed digest verification: {source}")
    _make_tree_durable(staging)
    os.rename(staging, destination)
    _fsync_parent_directory(destination)
    _digest, _size, files = _tree_digest(destination)
    return files


def _optional_tree_digest(path: Path) -> str | None:
    return _tree_digest(path)[0] if path.is_dir() else None


def _source_kind(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.casefold()
    if suffix == ".zip":
        return "zip"
    if suffix == ".rar":
        return "rar"
    raise RuntimeError(
        f"BoT-IoT source must be a local directory, ZIP, or RAR archive: {path}"
    )


def _reject_excluded_capture_files(root: Path) -> None:
    excluded = tuple(
        item
        for item in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix())
        if item.is_file() and item.suffix.casefold() in {".pcap", ".pcapng"}
    )
    if excluded:
        sample = ", ".join(str(item) for item in excluded[:3])
        raise RuntimeError(
            "PCAP capture input is excluded from the CSV bootstrap. Supply the "
            f"official model-ready CSV distribution instead; found: {sample}"
        )


def _extract_archive(
    source: Path,
    destination: Path,
    *,
    kind: str,
    dependencies: BootstrapDependencies,
    install_system_tools: bool,
) -> ExtractionResult:
    if kind == "zip":
        return dependencies.zip_extractor(source, destination)
    if kind == "rar":
        return dependencies.rar_extractor(
            source,
            destination,
            install_system_tools=install_system_tools,
        )
    raise RuntimeError(f"Unsupported archive kind {kind!r}")


def _nested_rars(root: Path) -> tuple[Path, ...]:
    return tuple(
        item
        for item in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix())
        if item.is_file() and item.suffix.casefold() == ".rar"
    )


def _recovery(stage: str | None) -> str:
    if stage is None or stage not in STAGE_ORDER:
        return (
            "Inspect the bootstrap report, correct the input or resource error, and "
            "rerun the original command."
        )
    return (
        "Inspect retained artifacts and correct the reported cause, then rerun the "
        f"original command with --restart-stage {stage}."
    )


def run_bootstrap(
    options: BootstrapOptions,
    *,
    raw_inputs: Mapping[str, object] | None = None,
    dependencies: BootstrapDependencies | None = None,
) -> dict[str, object]:
    """Acquire and verify selected CSV sources, returning a durable report."""

    deps = dependencies or BootstrapDependencies()
    preparation_enabled = dependencies is None or deps.preparer is not None
    metadata_root = options.data_root / ".bitguard"
    report_path = metadata_root / "bootstrap-report.json"
    state_path = metadata_root / "bootstrap-state.json"
    preflight_path = metadata_root / "preflight.json"
    environment_path = metadata_root / "environment.json"
    acquisition_report_path = metadata_root / "acquisition.json"
    extraction_report_path = metadata_root / "extraction.json"
    acquisition_journal_root = metadata_root / "acquisition-journal"
    extraction_journal_root = metadata_root / "extraction-journal"
    acquisition_root = metadata_root / "acquired"
    extraction_root = options.data_root / "raw"
    manifest_root = metadata_root / "manifests"
    schema_root = metadata_root / "schema"
    prepared_descriptor_root = metadata_root / "prepared"
    preparation_report_path = metadata_root / "preparation.json"
    preparation_work_root = metadata_root / "preparation-work"
    prepared_output_root = options.data_root / "prepared"
    original = {
        "botiot_source": None,
        "data_root": str(options.data_root),
        "runs_root": str(options.runs_root),
        **dict(raw_inputs or {}),
    }
    executed: list[str] = []
    reused: list[str] = []
    failed_stage: str | None = None
    last_completed: str | None = None
    report_persistence_error: BaseException | None = None
    lock_release_error: BaseException | None = None
    compute: dict[str, object] | None = None
    manifests: dict[str, str] = {}
    schemas: dict[str, str] = {}
    acquisition_journals: dict[str, str] = {}
    extraction_journals: dict[str, str] = {}
    prepared_datasets: dict[str, str] = {}
    cleanup_roots: set[Path] = {
        metadata_root,
        options.data_root,
        acquisition_root,
        acquisition_journal_root,
        extraction_root,
        extraction_journal_root,
        manifest_root,
        schema_root,
        prepared_descriptor_root,
        preparation_work_root,
        prepared_output_root,
    }

    def existing_locator(path: Path) -> str | None:
        try:
            return str(path) if path.is_file() else None
        except OSError:
            return None

    def cleanup_debt() -> dict[str, object]:
        try:
            return dict(scan_cleanup_debt(tuple(sorted(cleanup_roots, key=str))))
        except BaseException as error:
            return {
                "artifacts": [],
                "apparent_bytes": 0,
                "unique_bytes": 0,
                "recovery_command": inspection_command(
                    str(root) for root in sorted(cleanup_roots, key=str)
                ),
                "scan_errors": [
                    {"path": "<cleanup-roots>", "error": _safe_error(error)}
                ],
            }

    def build_report(
        status: str,
        error: BaseException | None,
        actual_report_path: Path,
    ) -> dict[str, object]:
        reports: dict[str, object] = {
            "bootstrap": str(actual_report_path),
            "preflight": existing_locator(preflight_path),
            "environment": existing_locator(environment_path),
            "acquisition": existing_locator(acquisition_report_path),
            "extraction": existing_locator(extraction_report_path),
            "schemas": dict(schemas),
            "acquisition_journals": dict(acquisition_journals),
            "extraction_journals": dict(extraction_journals),
        }
        if preparation_enabled:
            reports["preparation"] = existing_locator(preparation_report_path)
        result: dict[str, object] = {
            "version": REPORT_FORMAT_VERSION,
            "status": status,
            "last_completed_stage": last_completed,
            "failed_stage": failed_stage,
            "error": None if error is None else _safe_error(error),
            "report_error": (
                None
                if report_persistence_error is None
                else _safe_error(report_persistence_error)
            ),
            "lock_release_error": (
                None if lock_release_error is None else _safe_error(lock_release_error)
            ),
            "recovery_command": _recovery(failed_stage) if error is not None else None,
            "inputs": {"original": original, "resolved": options.to_dict()},
            "compute": compute,
            "state": str(state_path),
            "manifests": dict(manifests),
            "schema_reports": dict(schemas),
            "cleanup_debt": cleanup_debt(),
            "executed_stages": list(executed),
            "reused_stages": list(reused),
            "threat_model": (
                "Defends untrusted network/archive input and cooperative writers in a "
                "trusted workspace. Malicious same-account parent-namespace or hardlink "
                "mutation is outside this contract."
            ),
            "report_path": str(actual_report_path),
            "reports": reports,
        }
        if preparation_enabled:
            result["prepared_datasets"] = dict(prepared_datasets)
            result["next_stage"] = "train" if status == "prepared" else None
        redacted = _redact_report_value(result)
        if not isinstance(redacted, dict):  # pragma: no cover - fixed report shape
            raise TypeError("bootstrap report redaction produced a non-mapping")
        return redacted

    def persist_report(
        status: str,
        error: BaseException | None,
        *,
        preferred: Path | None,
    ) -> dict[str, object]:
        nonlocal failed_stage, report_persistence_error
        targets = ([preferred] if preferred is not None else []) + list(
            _fallback_report_paths(options)
        )
        unique_targets: list[Path] = []
        for target in targets:
            if target not in unique_targets:
                unique_targets.append(target)

        report_status = status
        primary_error = error
        last_write_error: BaseException | None = None
        for target in unique_targets:
            payload = build_report(report_status, primary_error, target)
            try:
                _write_json(target, payload)
                return payload
            except BaseException as write_error:
                last_write_error = write_error
                if report_persistence_error is None:
                    if (
                        isinstance(write_error, AtomicJsonWriteError)
                        and write_error.published
                    ):
                        report_persistence_error = RuntimeError(
                            "Bootstrap report directory durability is uncertain after "
                            "publication; a canonical failure-envelope reconciliation "
                            f"was required: {_safe_error(write_error)}"
                        )
                    else:
                        report_persistence_error = write_error
                report_status = "failed"
                if primary_error is None:
                    failed_stage = "report"
                    primary_error = RuntimeError(
                        "Bootstrap report persistence failed: "
                        f"{_safe_error(write_error)}"
                    )

                if not (
                    isinstance(write_error, AtomicJsonWriteError)
                    and write_error.published
                ):
                    continue

                failed_payload = build_report(report_status, primary_error, target)
                try:
                    _write_json(target, failed_payload)
                    return failed_payload
                except BaseException as reconciliation_error:
                    last_write_error = reconciliation_error
                    if (
                        isinstance(reconciliation_error, AtomicJsonWriteError)
                        and reconciliation_error.published
                    ):
                        # Replace succeeded, so the visible canonical artifact already
                        # carries the same failed status used by any fallback report.
                        continue

                    # Reconciliation did not replace the original success envelope.
                    # Invalidate only the exact writer-owned inode before publishing a
                    # fallback; a malformed canonical artifact cannot contradict it.
                    invalidation_error: BaseException | None = None
                    for _attempt in range(2):
                        try:
                            _invalidate_published_json(
                                target,
                                write_error.published_identity,
                            )
                            invalidation_error = None
                            break
                        except BaseException as invalidate_error:
                            invalidation_error = invalidate_error
                    if invalidation_error is not None:
                        try:
                            visible = json.loads(target.read_text(encoding="utf-8"))
                        except (
                            OSError,
                            UnicodeError,
                            json.JSONDecodeError,
                        ) as verification_error:
                            raise RuntimeError(
                                "Could not verify canonical bootstrap report state at "
                                f"{target} after invalidation failed; refusing fallback "
                                f"publication: {_safe_error(verification_error)}"
                            ) from verification_error
                        if (
                            isinstance(visible, dict)
                            and visible.get("status")
                            in {"sources_verified", "prepared"}
                        ):
                            raise RuntimeError(
                                "Could not invalidate a contradictory canonical "
                                f"bootstrap success report at {target}: "
                                f"{_safe_error(invalidation_error)}"
                            ) from invalidation_error
        assert last_write_error is not None
        return build_report(report_status, primary_error, unique_targets[-1])

    current_stage: str | None = "preflight"
    lock_entered = False
    try:
        lock_path = _lock_path(options)
    except BaseException as error:
        failed_stage = "preflight"
        return persist_report("failed", error, preferred=None)

    primary_error: BaseException | None = None
    final_status = "failed"
    lock: BootstrapWriterLock | None = None
    try:
        lock = BootstrapWriterLock(lock_path)
        lock.acquire()
        try:
            lock_entered = True
            try:
                current_stage = "preflight"
                registry = load_registry()
                _resolved_local_path(options.runs_root, "runs_root")
                source_context: dict[str, SourceContext] = {}
                if "nbaiot" in options.datasets:
                    if deps.nbaiot_archive is None:
                        remote_bytes = (
                            deps.nbaiot_download_size or UNKNOWN_REMOTE_ARCHIVE_BYTES
                        )
                        source_context["nbaiot"] = {
                            "kind": "zip",
                            "digest": None,
                            "bytes": remote_bytes,
                            "source": None,
                        }
                    else:
                        source = deps.nbaiot_archive.expanduser().resolve(strict=True)
                        if source.suffix.casefold() != ".zip":
                            raise RuntimeError(
                                "Injected N-BaIoT source must be a ZIP archive"
                            )
                        digest, size = _regular_digest(source)
                        source_context["nbaiot"] = {
                            "kind": "zip",
                            "digest": digest,
                            "bytes": size,
                            "source": source,
                        }
                if "botiot" in options.datasets:
                    if (
                        not options.accepted_botiot_license
                        or options.botiot_source is None
                    ):
                        raise RuntimeError(
                            "BoT-IoT requires a local source and explicit "
                            "academic-license acknowledgement"
                        )
                    _resolved_local_path(options.botiot_source, "botiot_source")
                    source = options.botiot_source.resolve(strict=True)
                    try:
                        source.relative_to(options.data_root)
                    except ValueError:
                        pass
                    else:
                        raise RuntimeError(
                            "BoT-IoT source must be outside the bootstrap data root"
                        )
                    try:
                        source.relative_to(options.runs_root)
                    except ValueError:
                        pass
                    else:
                        raise RuntimeError(
                            "BoT-IoT source must be outside the bootstrap runs root"
                        )
                    kind = _source_kind(source)
                    if kind == "directory":
                        _reject_excluded_capture_files(source)
                        digest, size, _files = _tree_digest(source)
                    else:
                        digest, size = _regular_digest(source)
                    source_context["botiot"] = {
                        "kind": kind,
                        "digest": digest,
                        "bytes": size,
                        "source": source,
                    }

                source_fingerprints: dict[str, str] = {}
                for dataset, context in source_context.items():
                    source_digest = context["digest"]
                    source_fingerprints[dataset] = (
                        source_digest
                        if isinstance(source_digest, str)
                        else _json_signature(
                            {
                                "dataset": dataset,
                                "registry": registry[dataset].to_dict(),
                            }
                        )
                    )

                download_bytes = sum(item["bytes"] for item in source_context.values())
                archive_bytes = sum(
                    item["bytes"]
                    for item in source_context.values()
                    if item["kind"] in {"zip", "rar"}
                )
                directory_bytes = download_bytes - archive_bytes
                estimate = estimate_resources(
                    ArchiveInspection(()),
                    final_download_bytes=download_bytes,
                    planned_partial_bytes=(
                        source_context["nbaiot"]["bytes"]
                        if deps.nbaiot_archive is None and "nbaiot" in source_context
                        else 0
                    ),
                    extracted_bytes=archive_bytes * ARCHIVE_EXPANSION_FACTOR
                    + directory_bytes,
                    shards_bytes=0,
                    evaluation_bytes=0,
                    temporary_bytes=REPORT_AND_METADATA_BYTES,
                    reserve_bytes=DEFAULT_DISK_RESERVE_BYTES,
                )
                if deps.available_bytes is None:
                    available = shutil.disk_usage(
                        _existing_parent(options.data_root)
                    ).free
                else:
                    available = deps.available_bytes
                disk = require_disk(estimate.request, available_bytes=available)

                _ensure_durable_directory(options.data_root)
                _ensure_durable_directory(metadata_root)
                _ensure_durable_directory(options.runs_root)
                state = BootstrapStateStore(state_path)
                if options.restart_stage is not None:
                    state.invalidate_from(options.restart_stage, STAGE_ORDER)

                acquired: dict[str, Path] = {}
                raw_roots: dict[str, Path] = {}
                for dataset, context in source_context.items():
                    token = source_fingerprints[dataset]
                    kind = str(context["kind"])
                    acquired[dataset] = (
                        acquisition_root / f"{dataset}-{token}"
                        if kind == "directory"
                        else acquisition_root / f"{dataset}-{token}.{kind}"
                    )
                    raw_roots[dataset] = extraction_root / f"{dataset}-{token}"

                def run_preflight() -> Sequence[Path]:
                    _write_json(
                        preflight_path,
                        {
                            "resources": estimate.as_dict(),
                            "disk": disk.as_dict(),
                            "remote_size_fallback_bytes": (
                                UNKNOWN_REMOTE_ARCHIVE_BYTES
                                if deps.nbaiot_archive is None
                                and deps.nbaiot_download_size is None
                                and "nbaiot" in source_context
                                else None
                            ),
                            "note": (
                                "Fresh remote acquisition reserves the complete partial "
                                "plus a complete verified final/snapshot. Unknown remote "
                                "size uses a conservative 4 GiB archive fallback."
                            ),
                        },
                    )
                    return (preflight_path,)

                def environment_payload() -> dict[str, object]:
                    nonlocal compute
                    if compute is None:
                        resolved_compute = dict(deps.compute_resolver(options.compute))
                        compute = _environment_manifest(options, resolved_compute)
                    return compute

                def run_environment() -> Sequence[Path]:
                    _write_json(environment_path, environment_payload())
                    return (environment_path,)

                def run_acquire() -> Sequence[Path]:
                    _ensure_durable_directory(acquisition_root)
                    _ensure_durable_directory(acquisition_journal_root)
                    outputs: list[Path] = []
                    acquisition_report: dict[str, object] = {}
                    journal_fields = {
                        "version",
                        "dataset",
                        "status",
                        "kind",
                        "source_sha256",
                        "final_path",
                        "candidate_path",
                        "content_sha256",
                        "content_bytes",
                        "report",
                    }
                    for dataset, context in source_context.items():
                        destination = acquired[dataset]
                        source = context["source"]
                        digest = context["digest"]
                        source_fingerprint = source_fingerprints[dataset]
                        kind = str(context["kind"])
                        journal_path = acquisition_journal_root / f"{dataset}.json"
                        record = _load_journal(journal_path, f"{dataset} acquisition")
                        current_record = False
                        if record is not None:
                            _require_journal_shape(
                                record,
                                subject=f"{dataset} acquisition",
                                fields=journal_fields,
                            )
                            if not isinstance(record["kind"], str):
                                raise RuntimeError(
                                    f"{dataset} acquisition journal has an invalid kind"
                                )
                            source_sha256 = record["source_sha256"]
                            if source_sha256 is not None and not isinstance(
                                source_sha256, str
                            ):
                                raise RuntimeError(
                                    f"{dataset} acquisition journal has an invalid "
                                    "source_sha256"
                                )
                            if not isinstance(
                                record["content_bytes"], int
                            ) or isinstance(record["content_bytes"], bool):
                                raise RuntimeError(
                                    f"{dataset} acquisition journal has invalid content_bytes"
                                )
                            safe_report = _sanitized_report_mapping(
                                record["report"],
                                f"{dataset} acquisition journal",
                            )
                            if safe_report != record["report"]:
                                record = {**record, "report": safe_report}
                                _write_json(journal_path, record)
                            current_record = (
                                record["dataset"] == dataset
                                and record["kind"] == kind
                                and record["source_sha256"] == source_fingerprint
                                and record["final_path"] == str(destination)
                            )

                        if current_record:
                            assert record is not None
                            expected_fingerprint = (
                                str(record["content_sha256"]),
                                cast(int, record["content_bytes"]),
                            )
                            candidate_value = record["candidate_path"]
                            retained = (
                                _retained_candidate(
                                    candidate_value,
                                    destination=destination,
                                    prefix=f".bitguard-acquire-{dataset}-",
                                )
                                if isinstance(candidate_value, str)
                                else None
                            )
                            if destination.exists():
                                actual = _content_fingerprint(destination, kind)
                                if actual != expected_fingerprint:
                                    raise RuntimeError(
                                        "Existing acquisition output does not match its "
                                        f"dataset journal: {destination}. Preserve it for "
                                        "inspection and use a new data root."
                                    )
                                if record["status"] == "intent":
                                    record = {
                                        **record,
                                        "status": "completed",
                                        "candidate_path": (
                                            str(retained)
                                            if kind != "directory"
                                            and retained is not None
                                            else None
                                        ),
                                    }
                                    _write_json(journal_path, record)
                                acquisition_journals[dataset] = str(journal_path)
                                acquisition_report[dataset] = dict(
                                    cast(dict[str, object], record["report"])
                                )
                                outputs.extend(_published_outputs(destination, kind))
                                continue

                            if retained is not None:
                                actual = _content_fingerprint(retained, kind)
                                if actual != expected_fingerprint:
                                    raise RuntimeError(
                                        "Retained acquisition candidate does not match "
                                        f"its dataset journal: {retained}."
                                    )
                                _publish_candidate(retained, destination, kind)
                                record = {
                                    **record,
                                    "status": "completed",
                                    "candidate_path": (
                                        str(retained) if kind != "directory" else None
                                    ),
                                }
                                _write_json(journal_path, record)
                                acquisition_journals[dataset] = str(journal_path)
                                acquisition_report[dataset] = dict(
                                    cast(dict[str, object], record["report"])
                                )
                                outputs.extend(_published_outputs(destination, kind))
                                continue

                        if destination.exists():
                            raise RuntimeError(
                                "Existing acquisition output has no matching verified "
                                f"dataset journal: {destination}. Preserve it for inspection "
                                "and use a new data root."
                            )

                        candidate = destination.parent / (
                            f".bitguard-acquire-{dataset}-{uuid.uuid4().hex}"
                        )
                        if dataset == "nbaiot" and source is None:
                            spec = registry[dataset]
                            assert spec.download_url is not None
                            prior_hash = (
                                str(record["content_sha256"])
                                if current_record and record is not None
                                else None
                            )
                            result = deps.downloader(
                                spec.download_url,
                                candidate,
                                expected_sha256=prior_hash,
                            )
                            dataset_report = result.to_dict()
                            dataset_report["destination"] = str(destination)
                        elif kind == "directory":
                            assert isinstance(source, Path) and isinstance(digest, str)
                            _copy_tree(source, candidate, digest)
                            dataset_report = {
                                "method": "manual-local-source",
                                "source": str(source),
                                "destination": str(destination),
                                "sha256": digest,
                            }
                        else:
                            assert isinstance(source, Path) and isinstance(digest, str)
                            _copy_file(source, candidate, digest)
                            dataset_report = {
                                "method": (
                                    "official-download-fixture"
                                    if dataset == "nbaiot"
                                    else "manual-local-source"
                                ),
                                "source": str(source),
                                "destination": str(destination),
                                "sha256": digest,
                            }
                        _make_content_durable(candidate, kind)
                        dataset_report = _sanitized_report_mapping(
                            dataset_report,
                            f"{dataset} acquisition",
                        )
                        fingerprint, content_bytes = _content_fingerprint(
                            candidate, kind
                        )
                        intent: dict[str, object] = {
                            "version": JOURNAL_FORMAT_VERSION,
                            "dataset": dataset,
                            "status": "intent",
                            "kind": kind,
                            "source_sha256": source_fingerprint,
                            "final_path": str(destination),
                            "candidate_path": str(candidate),
                            "content_sha256": fingerprint,
                            "content_bytes": content_bytes,
                            "report": dataset_report,
                        }
                        _write_json(journal_path, intent)
                        acquisition_journals[dataset] = str(journal_path)
                        _publish_candidate(candidate, destination, kind)
                        if _content_fingerprint(destination, kind) != (
                            fingerprint,
                            content_bytes,
                        ):
                            raise RuntimeError(
                                f"Published acquisition output failed journal validation: "
                                f"{destination}"
                            )
                        completed = {
                            **intent,
                            "status": "completed",
                            "candidate_path": (
                                str(candidate) if kind != "directory" else None
                            ),
                        }
                        _write_json(journal_path, completed)
                        acquisition_report[dataset] = dataset_report
                        outputs.extend(_published_outputs(destination, kind))
                    _write_json(
                        acquisition_report_path, {"datasets": acquisition_report}
                    )
                    outputs.append(acquisition_report_path)
                    return tuple(outputs)

                def run_extract() -> Sequence[Path]:
                    _ensure_durable_directory(extraction_root)
                    _ensure_durable_directory(extraction_journal_root)
                    outputs: list[Path] = []
                    extraction_report: dict[str, object] = {}
                    journal_fields = {
                        "version",
                        "dataset",
                        "status",
                        "source_sha256",
                        "final_path",
                        "candidate_path",
                        "content_sha256",
                        "tree_sha256",
                        "report",
                    }
                    for dataset, context in source_context.items():
                        source = acquired[dataset]
                        destination = raw_roots[dataset]
                        kind = str(context["kind"])
                        source_sha256 = _content_fingerprint(source, kind)[0]
                        journal_path = extraction_journal_root / f"{dataset}.json"
                        record = _load_journal(journal_path, f"{dataset} extraction")
                        current_record = False
                        if record is not None:
                            _require_journal_shape(
                                record,
                                subject=f"{dataset} extraction",
                                fields=journal_fields,
                            )
                            if not isinstance(record["source_sha256"], str):
                                raise RuntimeError(
                                    f"{dataset} extraction journal has an invalid "
                                    "source_sha256"
                                )
                            if (
                                not isinstance(record["tree_sha256"], str)
                                or not record["tree_sha256"]
                            ):
                                raise RuntimeError(
                                    f"{dataset} extraction journal has an invalid "
                                    "tree_sha256"
                                )
                            if record["tree_sha256"] != record["content_sha256"]:
                                raise RuntimeError(
                                    f"{dataset} extraction journal has inconsistent "
                                    "tree digests"
                                )
                            safe_report = _sanitized_report_mapping(
                                record["report"],
                                f"{dataset} extraction journal",
                            )
                            if safe_report != record["report"]:
                                record = {**record, "report": safe_report}
                                _write_json(journal_path, record)
                            current_record = (
                                record["dataset"] == dataset
                                and record["source_sha256"] == source_sha256
                                and record["final_path"] == str(destination)
                            )

                        if current_record:
                            assert record is not None
                            expected_tree = str(record["tree_sha256"])
                            candidate_value = record["candidate_path"]
                            retained = (
                                _retained_candidate(
                                    candidate_value,
                                    destination=destination,
                                    prefix=f".bitguard-extract-{dataset}-",
                                )
                                if isinstance(candidate_value, str)
                                else None
                            )
                            if destination.exists():
                                actual_tree = _tree_digest(destination)[0]
                                if actual_tree != expected_tree:
                                    raise RuntimeError(
                                        "Existing extracted source does not match its "
                                        "prior verified tree recorded in its dataset "
                                        f"journal: {destination}. Preserve it for inspection "
                                        "and use a new data root."
                                    )
                                _reject_excluded_capture_files(destination)
                                if record["status"] == "intent":
                                    record = {
                                        **record,
                                        "status": "completed",
                                        "candidate_path": None,
                                    }
                                    _write_json(journal_path, record)
                                extraction_journals[dataset] = str(journal_path)
                                extraction_report[dataset] = dict(
                                    cast(dict[str, object], record["report"])
                                )
                                continue

                            if retained is not None:
                                actual_tree = _tree_digest(retained)[0]
                                if actual_tree != expected_tree:
                                    raise RuntimeError(
                                        "Retained extraction candidate does not match "
                                        f"its dataset journal: {retained}."
                                    )
                                _reject_excluded_capture_files(retained)
                                _publish_candidate(
                                    retained,
                                    destination,
                                    "directory",
                                )
                                record = {
                                    **record,
                                    "status": "completed",
                                    "candidate_path": None,
                                }
                                _write_json(journal_path, record)
                                extraction_journals[dataset] = str(journal_path)
                                extraction_report[dataset] = dict(
                                    cast(dict[str, object], record["report"])
                                )
                                continue

                        if destination.exists():
                            if destination.is_dir():
                                _reject_excluded_capture_files(destination)
                                raise RuntimeError(
                                    "Existing extracted source has no matching verified "
                                    f"dataset journal: {destination}. Preserve it for "
                                    "inspection and use a new data root."
                                )
                            raise RuntimeError(
                                f"Extraction destination is not a directory: {destination}"
                            )

                        candidate = destination.with_name(
                            f".bitguard-extract-{dataset}-{uuid.uuid4().hex}"
                        )
                        if kind == "directory":
                            copied = _copy_tree(source, candidate, source_sha256)
                            dataset_report: dict[str, object] = {
                                "extractor": "verified-directory-copy",
                                "destination": str(destination),
                                "files": len(copied),
                            }
                        else:
                            result = _extract_archive(
                                source,
                                candidate,
                                kind=kind,
                                dependencies=deps,
                                install_system_tools=options.install_system_tools,
                            )
                            nested_results: list[ExtractionResult] = []
                            for nested in _nested_rars(candidate):
                                nested_destination = nested.with_suffix("")
                                nested_result = deps.rar_extractor(
                                    nested,
                                    nested_destination,
                                    install_system_tools=options.install_system_tools,
                                )
                                nested_results.append(nested_result)
                            _reject_excluded_capture_files(candidate)

                            def relocated(value: ExtractionResult) -> dict[str, object]:
                                payload = value.as_dict()
                                for field in ("source", "destination"):
                                    path = Path(str(payload[field]))
                                    try:
                                        relative = path.relative_to(candidate)
                                    except ValueError:
                                        continue
                                    payload[field] = str(destination / relative)
                                return payload

                            dataset_report = {
                                **relocated(result),
                                "nested_rar": [
                                    relocated(nested_result)
                                    for nested_result in nested_results
                                ],
                            }
                        dataset_report = _sanitized_report_mapping(
                            dataset_report,
                            f"{dataset} extraction",
                        )
                        _reject_excluded_capture_files(candidate)
                        _make_tree_durable(candidate)
                        tree_sha256 = _tree_digest(candidate)[0]
                        dataset_report["tree_sha256"] = tree_sha256
                        intent: dict[str, object] = {
                            "version": JOURNAL_FORMAT_VERSION,
                            "dataset": dataset,
                            "status": "intent",
                            "source_sha256": source_sha256,
                            "final_path": str(destination),
                            "candidate_path": str(candidate),
                            "content_sha256": tree_sha256,
                            "tree_sha256": tree_sha256,
                            "report": dataset_report,
                        }
                        _write_json(journal_path, intent)
                        extraction_journals[dataset] = str(journal_path)
                        try:
                            _publish_candidate(candidate, destination, "directory")
                        except FileExistsError as error:
                            raise RuntimeError(
                                "Final extraction destination appeared before "
                                f"publication: {destination}; private candidate "
                                f"is retained at {candidate}."
                            ) from error
                        except OSError as error:
                            raise RuntimeError(
                                "Could not publish verified extraction candidate "
                                f"{candidate} to {destination}: {error}. The "
                                "private candidate is retained for inspection."
                            ) from error
                        if _tree_digest(destination)[0] != tree_sha256:
                            raise RuntimeError(
                                "Published extraction output failed journal validation: "
                                f"{destination}"
                            )
                        completed = {
                            **intent,
                            "status": "completed",
                            "candidate_path": None,
                        }
                        _write_json(journal_path, completed)
                        extraction_report[dataset] = dataset_report
                        _reject_excluded_capture_files(destination)
                    _write_json(extraction_report_path, {"datasets": extraction_report})
                    outputs.append(extraction_report_path)
                    return tuple(outputs)

                def run_inspect() -> Sequence[Path]:
                    _ensure_durable_directory(manifest_root)
                    _ensure_durable_directory(schema_root)
                    outputs: list[Path] = []
                    for dataset in options.datasets:
                        spec: DatasetSpec = registry[dataset]
                        raw_root = raw_roots[dataset]
                        source_token = _tree_digest(raw_root)[0]
                        manifest = build_source_manifest(
                            raw_root,
                            spec,
                            acquisition_method=(
                                "official-download"
                                if dataset == "nbaiot"
                                else "manual-local-source"
                            ),
                            acquisition_url=(
                                spec.download_url if dataset == "nbaiot" else None
                            ),
                        )
                        manifest_path = manifest_root / f"{dataset}-{source_token}.json"
                        write_source_manifest(manifest_path, manifest)
                        schema = deps.inspector(
                            dataset,
                            raw_root,
                            required_columns=spec.required_columns,
                        )
                        schema_path = schema_root / f"{dataset}-{source_token}.json"
                        _write_json(schema_path, schema.as_dict())
                        manifests[dataset] = str(manifest_path)
                        schemas[dataset] = str(schema_path)
                        outputs.extend((manifest_path, schema_path))
                    return tuple(outputs)

                def full_config_path(dataset: str) -> Path:
                    repository = Path(__file__).resolve().parents[3]
                    return repository / "configs" / "full" / f"{dataset}.yaml"

                def prepared_descriptor_path(dataset: str) -> Path:
                    generation = preparation_generation(dataset)
                    return prepared_descriptor_root / dataset / f"{generation}.json"

                def prepared_generation_output(dataset: str) -> Path:
                    return prepared_output_root / dataset / preparation_generation(dataset)

                def prepared_generation_work(dataset: str) -> Path:
                    return preparation_work_root / dataset / preparation_generation(dataset)

                preparation_signature_cache: dict[str, str] = {}
                preparation_contract_cache: dict[str, object] | None = None

                def preparation_contract(*, refresh: bool = False) -> dict[str, object]:
                    nonlocal preparation_contract_cache
                    if preparation_contract_cache is not None and not refresh:
                        return preparation_contract_cache
                    from bitguard_bnn.out_of_core import (
                        prepare as prepare_module,
                        preprocess as preprocess_module,
                        quantiles as quantiles_module,
                        shard as shard_module,
                        source as source_module,
                        split as split_module,
                    )

                    contract = {
                        **prepare_module.preparation_implementation_contract(),
                        "algorithm_versions": {
                            "source_normalization": "bitguard.normalization-signature.v1",
                            "split": split_module.SPLIT_ALGORITHM,
                            "split_membership": split_module.MEMBERSHIP_ALGORITHM,
                            "quantile": (
                                f"{quantiles_module.ALGORITHM}.v"
                                f"{quantiles_module.VERSION}"
                            ),
                            "preprocess": preprocess_module.STREAMING_PREPROCESS_VERSION,
                            "shard": shard_module.SHARD_ALGORITHM,
                            "coverage": shard_module.COVERAGE_ALGORITHM,
                        },
                        "dependency_token": deps.preparation_signature_token,
                        "environment": _regular_digest(environment_path)[0],
                    }
                    if not refresh:
                        preparation_contract_cache = contract
                    return contract

                def dataset_preparation_signature(
                    dataset: str,
                    *,
                    refresh: bool = False,
                    contract: Mapping[str, object] | None = None,
                ) -> str:
                    cached = preparation_signature_cache.get(dataset)
                    if cached is not None and not refresh:
                        return cached
                    signature = _json_signature(
                        {
                            "dataset": dataset,
                            "inputs": {
                                "raw": _tree_digest(raw_roots[dataset])[0],
                                "source_manifest": _regular_digest(
                                    Path(manifests[dataset])
                                )[0],
                                "schema": _regular_digest(Path(schemas[dataset]))[0],
                                "config": _regular_digest(full_config_path(dataset))[0],
                            },
                            "contract": (
                                contract
                                if contract is not None
                                else preparation_contract(refresh=refresh)
                            ),
                        }
                    )
                    if not refresh:
                        preparation_signature_cache[dataset] = signature
                    return signature

                def preparation_signature(*, refresh: bool = False) -> str:
                    refreshed_contract = (
                        preparation_contract(refresh=True) if refresh else None
                    )
                    return _json_signature(
                        {
                            dataset: dataset_preparation_signature(
                                dataset,
                                refresh=refresh,
                                contract=refreshed_contract,
                            )
                            for dataset in options.datasets
                        }
                    )

                def preparation_generation(dataset: str) -> str:
                    return dataset_preparation_signature(dataset)

                def preparation_disk_requirements() -> dict[str, object]:
                    from bitguard_bnn.out_of_core.prepare import (
                        estimate_preparation_disk,
                        verify_prepared_dataset,
                    )

                    verifier = deps.prepared_verifier or verify_prepared_dataset
                    pending: list[str] = []
                    verified_existing: list[str] = []
                    for dataset in options.datasets:
                        descriptor = prepared_descriptor_path(dataset)
                        try:
                            descriptor_stat = descriptor.lstat()
                        except FileNotFoundError:
                            pending.append(dataset)
                            continue
                        if (
                            stat.S_ISLNK(descriptor_stat.st_mode)
                            or bool(getattr(descriptor_stat, "st_reparse_tag", 0))
                            or not stat.S_ISREG(descriptor_stat.st_mode)
                        ):
                            raise RuntimeError(
                                "existing prepared descriptor is not a regular file: "
                                f"{descriptor}"
                            )
                        verifier(descriptor)
                        prepared_datasets[dataset] = str(descriptor)
                        verified_existing.append(dataset)

                    estimate_objects: dict[str, object] = {}
                    estimates: dict[str, dict[str, int]] = {}
                    for dataset in pending:
                        estimate = estimate_preparation_disk(
                            Path(manifests[dataset]),
                            Path(schemas[dataset]),
                            train_fraction=0.70,
                        )
                        estimate_objects[dataset] = estimate
                        estimates[dataset] = estimate.as_dict()
                    groups = _group_preparation_disk_requirements(
                        estimate_objects,
                        work_paths={
                            dataset: prepared_generation_work(dataset)
                            for dataset in options.datasets
                        },
                        output_paths={
                            dataset: prepared_generation_output(dataset)
                            for dataset in options.datasets
                        },
                        device_for=lambda path: int(
                            _existing_parent(path).stat().st_dev
                        ),
                    )
                    for group in groups.values():
                        probe = _existing_parent(Path(str(group["path"])))
                        group["path"] = str(probe)
                        available_bytes = (
                            deps.preparation_available_bytes
                            if deps.preparation_available_bytes is not None
                            else deps.available_bytes
                            if deps.available_bytes is not None
                            else shutil.disk_usage(probe).free
                        )
                        required_bytes = cast(int, group["required_bytes"])
                        group["available_bytes"] = int(available_bytes)
                        if int(available_bytes) < required_bytes:
                            raise RuntimeError(
                                "Insufficient disk for full-data preparation on "
                                f"{probe}: required={required_bytes} bytes, "
                                f"available={available_bytes} bytes."
                            )
                    return {
                        "estimates": estimates,
                        "device_groups": {
                            str(device): value for device, value in sorted(groups.items())
                        },
                        "pending_datasets": pending,
                        "excluded_verified_datasets": verified_existing,
                    }

                preparation_disk: dict[str, object] = {}

                def run_shard() -> Sequence[Path]:
                    nonlocal preparation_disk
                    from bitguard_bnn.out_of_core.prepare import prepare_full_dataset

                    preparation_disk = preparation_disk_requirements()
                    _ensure_durable_directory(prepared_descriptor_root)
                    _ensure_durable_directory(prepared_output_root)
                    _ensure_durable_directory(preparation_work_root)
                    preparer = deps.preparer or prepare_full_dataset
                    outputs: list[Path] = []
                    pending = set(
                        cast(Sequence[str], preparation_disk["pending_datasets"])
                    )
                    for dataset in options.datasets:
                        descriptor = prepared_descriptor_path(dataset)
                        if dataset not in pending:
                            prepared_datasets[dataset] = str(descriptor)
                            outputs.append(descriptor)
                            continue
                        generation = preparation_generation(dataset)
                        preparer(
                            full_config_path(dataset),
                            raw_root=raw_roots[dataset],
                            source_manifest_path=Path(manifests[dataset]),
                            schema_report_path=Path(schemas[dataset]),
                            output_dir=prepared_generation_output(dataset),
                            descriptor_path=descriptor,
                            work_dir=prepared_generation_work(dataset),
                            preparation_signature=generation,
                        )
                        if not descriptor.is_file() or descriptor.is_symlink():
                            raise RuntimeError(
                                f"preparation did not publish its control descriptor: {descriptor}"
                            )
                        prepared_datasets[dataset] = str(descriptor)
                        outputs.append(descriptor)
                    return tuple(outputs)

                def validate_signature() -> str:
                    return _json_signature(
                        {
                            dataset: _regular_digest(
                                prepared_descriptor_path(dataset)
                            )[0]
                            for dataset in options.datasets
                        }
                    )

                def run_validate() -> Sequence[Path]:
                    from bitguard_bnn.out_of_core.prepare import verify_prepared_dataset

                    verifier = deps.prepared_verifier or verify_prepared_dataset
                    verified: dict[str, object] = {}
                    for dataset in options.datasets:
                        descriptor = prepared_descriptor_path(dataset)
                        result = verifier(descriptor)
                        prepared_datasets[dataset] = str(descriptor)
                        if hasattr(result, "to_dict"):
                            verified[dataset] = result.to_dict()
                        else:
                            verified[dataset] = {
                                "descriptor_path": str(descriptor),
                                "descriptor_sha256": _regular_digest(descriptor)[0],
                            }
                    if not preparation_disk:
                        preparation_disk.update(preparation_disk_requirements())
                    _write_json(
                        preparation_report_path,
                        {
                            "status": "prepared",
                            "datasets": verified,
                            "disk": dict(preparation_disk),
                            "prepare_only": options.prepare_only,
                            "next_stage": "train",
                        },
                    )
                    return (preparation_report_path,)

                stages: tuple[Stage, ...] = (
                    Stage(
                        "preflight",
                        lambda: _json_signature(
                            {
                                "resources": estimate.as_dict(),
                                "sources": {
                                    name: {
                                        "kind": context["kind"],
                                        "bytes": context["bytes"],
                                    }
                                    for name, context in source_context.items()
                                },
                            }
                        ),
                        run_preflight,
                    ),
                    Stage(
                        "environment",
                        lambda: _json_signature(environment_payload()),
                        run_environment,
                    ),
                    Stage(
                        "acquire",
                        lambda: _json_signature(
                            {
                                "sources": {
                                    name: {
                                        "digest": context["digest"],
                                        "kind": context["kind"],
                                        "bytes": context["bytes"],
                                    }
                                    for name, context in source_context.items()
                                },
                                "official_urls": {
                                    name: registry[name].download_url
                                    for name in options.datasets
                                },
                            }
                        ),
                        run_acquire,
                    ),
                    Stage(
                        "extract",
                        lambda: _json_signature(
                            {
                                "acquired": {
                                    name: (
                                        _regular_digest(path)[0]
                                        if path.is_file()
                                        else _tree_digest(path)[0]
                                    )
                                    for name, path in acquired.items()
                                },
                                "raw_outputs": {
                                    name: _optional_tree_digest(path)
                                    for name, path in raw_roots.items()
                                },
                                "install_system_tools": options.install_system_tools,
                            }
                        ),
                        run_extract,
                    ),
                    Stage(
                        "inspect",
                        lambda: _json_signature(
                            {
                                "raw": {
                                    name: _tree_digest(path)[0]
                                    for name, path in raw_roots.items()
                                },
                                "registry": {
                                    name: registry[name].to_dict()
                                    for name in options.datasets
                                },
                            }
                        ),
                        run_inspect,
                    ),
                )
                if preparation_enabled:
                    stages += (
                        Stage("shard", lambda: preparation_signature(), run_shard),
                        Stage(
                            "validate",
                            validate_signature,
                            run_validate,
                            always_run=True,
                        ),
                    )

                for stage in stages:
                    current_stage = stage.name
                    signature = stage.input_signature()
                    reusable = state.reusable(stage.name, signature)
                    if reusable and not stage.always_run:
                        reused.append(stage.name)
                    else:
                        if not reusable:
                            state.invalidate_from(stage.name, STAGE_ORDER)
                        outputs = tuple(stage.run())
                        completion_signature = stage.input_signature()
                        if stage.name == "shard":
                            completion_signature = preparation_signature(refresh=True)
                            if completion_signature != signature:
                                raise RuntimeError(
                                    "full-data preparation inputs changed while the "
                                    "shard stage was running"
                                )
                        state.complete(stage.name, completion_signature, outputs)
                        executed.append(stage.name)
                    last_completed = stage.name
                    if stage.name == "environment" and compute is None:
                        compute = json.loads(
                            environment_path.read_text(encoding="utf-8")
                        )
                    if stage.name == "inspect":
                        for dataset in options.datasets:
                            source_token = _tree_digest(raw_roots[dataset])[0]
                            manifests[dataset] = str(
                                manifest_root / f"{dataset}-{source_token}.json"
                            )
                            schemas[dataset] = str(
                                schema_root / f"{dataset}-{source_token}.json"
                            )
                    if stage.name == "shard" and reusable:
                        for dataset in options.datasets:
                            prepared_datasets[dataset] = str(
                                prepared_descriptor_path(dataset)
                            )

                final_status = (
                    "prepared" if preparation_enabled else "sources_verified"
                )
            except BaseException as error:
                failed_stage = current_stage
                primary_error = error
                final_status = "failed"
        finally:
            if lock_entered:
                assert lock is not None
                released = lock.release()
                if not released:
                    raise RuntimeError(
                        f"Bootstrap writer lock {lock_path} could not be released "
                        "with verified ownership."
                    )
    except BaseException as error:
        final_status = "failed"
        if lock_entered:
            lock_release_error = error
            if primary_error is None:
                failed_stage = "lock-release"
                primary_error = error
        else:
            failed_stage = "preflight"
            primary_error = error
    return persist_report(
        final_status,
        primary_error,
        preferred=report_path if lock_entered else None,
    )
