"""Read-only resource, platform, and compute checks for bootstrap orchestration."""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import math
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ResourceProbeError(RuntimeError):
    """A resource fact could not be established safely."""


class NvidiaProbeError(ResourceProbeError):
    """The NVIDIA probe returned an unsafe or ambiguous result."""


def _nonnegative_bytes(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer byte count, got {value!r}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceProbeError(f"{name} must be a positive integer, got {value!r}.")
    if value <= 0:
        raise ResourceProbeError(f"{name} must be positive, got {value}.")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResourceProbeError(f"{name} must be a non-empty string, got {value!r}.")
    return value.strip()


def _device_index(value: object, name: str = "device_index") -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer or None, got {value!r}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Conservative peak bootstrap disk usage, expressed only in integer bytes."""

    download: int
    extracted: int
    shards: int
    temporary: int
    reserve: int
    partial: int = 0
    evaluation: int = 0

    def __post_init__(self) -> None:
        for name in (
            "download",
            "extracted",
            "shards",
            "temporary",
            "reserve",
            "partial",
            "evaluation",
        ):
            _nonnegative_bytes(getattr(self, name), name)

    @property
    def required_bytes(self) -> int:
        return sum(self.breakdown().values())

    def breakdown(self) -> dict[str, int]:
        return {
            "final_downloads": self.download,
            "partial_downloads": self.partial,
            "extracted_data": self.extracted,
            "parquet_shards": self.shards,
            "evaluation_artifacts": self.evaluation,
            "temporary_workspace": self.temporary,
            "reserve": self.reserve,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "required_bytes": self.required_bytes,
            "breakdown": self.breakdown(),
        }


@dataclass(frozen=True, slots=True)
class DiskFacts:
    required_bytes: int
    available_bytes: int
    path: str | None = None

    def __post_init__(self) -> None:
        _nonnegative_bytes(self.required_bytes, "required_bytes")
        _nonnegative_bytes(self.available_bytes, "available_bytes")
        if self.path is not None and (not isinstance(self.path, str) or not self.path):
            raise TypeError("path must be a non-empty string or None.")

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "required_bytes": self.required_bytes,
            "available_bytes": self.available_bytes,
        }


def require_disk(request: ResourceRequest, *, available_bytes: int) -> DiskFacts:
    """Fail before mutation unless the supplied free-space observation is sufficient."""

    if not isinstance(request, ResourceRequest):
        raise TypeError("request must be a ResourceRequest.")
    available = _nonnegative_bytes(available_bytes, "available_bytes")
    required = request.required_bytes
    if available < required:
        shortfall = required - available
        raise RuntimeError(
            "Insufficient disk space before bootstrap mutation: "
            f"required={required} bytes, available={available} bytes, "
            f"shortfall={shortfall} bytes. Free disk space or reduce the explicit estimates."
        )
    return DiskFacts(required_bytes=required, available_bytes=available)


def require_disk_at(
    request: ResourceRequest,
    path: str | os.PathLike[str],
    *,
    disk_usage_fn: Callable[[Path], Any] | None = None,
) -> DiskFacts:
    """Read free space for an existing filesystem location without creating it."""

    resolved = Path(path).expanduser().resolve(strict=False)
    usage_probe = shutil.disk_usage if disk_usage_fn is None else disk_usage_fn
    try:
        usage = usage_probe(resolved)
        free = usage.free
    except Exception as error:
        raise ResourceProbeError(
            f"Could not inspect disk space for {resolved}: {error}. "
            "Use an existing path on the target filesystem."
        ) from error
    free_bytes = _nonnegative_bytes(free, f"disk free space for {resolved}")
    facts = require_disk(request, available_bytes=free_bytes)
    return DiskFacts(
        required_bytes=facts.required_bytes,
        available_bytes=facts.available_bytes,
        path=str(resolved),
    )


@dataclass(frozen=True, slots=True)
class ArchiveObservation:
    path: str
    size_bytes: int
    partial_path: str | None = None
    partial_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise TypeError("Archive observation path must be a non-empty string.")
        _nonnegative_bytes(self.size_bytes, "archive size_bytes")
        _nonnegative_bytes(self.partial_bytes, "archive partial_bytes")
        if self.partial_path is None and self.partial_bytes:
            raise ValueError("partial_bytes requires partial_path.")
        if self.partial_path is not None and (
            not isinstance(self.partial_path, str) or not self.partial_path
        ):
            raise TypeError("partial_path must be a non-empty string or None.")

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "partial_path": self.partial_path,
            "partial_bytes": self.partial_bytes,
        }


@dataclass(frozen=True, slots=True)
class ArchiveInspection:
    archives: tuple[ArchiveObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.archives, tuple) or not all(
            isinstance(item, ArchiveObservation) for item in self.archives
        ):
            raise TypeError("archives must be a tuple of ArchiveObservation values.")

    @property
    def total_archive_bytes(self) -> int:
        return sum(item.size_bytes for item in self.archives)

    @property
    def total_partial_bytes(self) -> int:
        return sum(item.partial_bytes for item in self.archives)

    def as_dict(self) -> dict[str, object]:
        return {
            "archives": [item.as_dict() for item in self.archives],
            "total_archive_bytes": self.total_archive_bytes,
            "total_partial_bytes": self.total_partial_bytes,
        }


def _snapshot_fields(path: Path, result: Any, kind: str) -> tuple[int, int, int, int, int]:
    try:
        mode = result.st_mode
        size = result.st_size
        modified = result.st_mtime_ns
        device = result.st_dev
        inode = result.st_ino
    except (AttributeError, TypeError) as error:
        raise ResourceProbeError(
            f"Cannot safely inspect {kind} {path}: stat result is incomplete."
        ) from error
    if isinstance(mode, bool) or not isinstance(mode, int) or not stat.S_ISREG(mode):
        raise ResourceProbeError(f"{kind.capitalize()} {path} is not a regular file.")
    size_bytes = _nonnegative_bytes(size, f"size of {kind} {path}")
    for value, name in (
        (modified, "modification time"),
        (device, "device identifier"),
        (inode, "inode identifier"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ResourceProbeError(
                f"Cannot safely inspect {kind} {path}: {name} is not an integer."
            )
    return mode, size_bytes, modified, device, inode


def _observe_regular_file(
    path: Path,
    *,
    kind: str,
    stat_fn: Callable[[Path], Any],
    allow_missing: bool,
) -> int | None:
    try:
        first_result = stat_fn(path)
    except FileNotFoundError as error:
        if allow_missing:
            try:
                stat_fn(path)
            except FileNotFoundError:
                return None
            except OSError as second_error:
                raise ResourceProbeError(
                    f"{kind.capitalize()} {path} changed during inspection: "
                    f"{second_error}."
                ) from second_error
            raise ResourceProbeError(
                f"{kind.capitalize()} {path} changed during inspection: "
                "it appeared after initially being absent."
            ) from error
        raise ResourceProbeError(f"Archive does not exist: {path}.") from error
    except OSError as error:
        raise ResourceProbeError(f"Cannot inspect {kind} {path}: {error}.") from error
    first = _snapshot_fields(path, first_result, kind)
    try:
        second_result = stat_fn(path)
    except (FileNotFoundError, OSError) as error:
        raise ResourceProbeError(
            f"{kind.capitalize()} {path} changed during inspection: {error}."
        ) from error
    second = _snapshot_fields(path, second_result, kind)
    if first != second:
        raise ResourceProbeError(
            f"{kind.capitalize()} {path} changed during inspection; retry once the file is stable."
        )
    return first[1]


def inspect_local_archives(
    paths: Iterable[str | os.PathLike[str]],
    *,
    stat_fn: Callable[[Path], Any] | None = None,
    lstat_fn: Callable[[Path], Any] | None = None,
) -> ArchiveInspection:
    """Resolve and stably size complete archives plus adjacent ``.partial`` files."""

    probe = (lambda candidate: candidate.stat()) if stat_fn is None else stat_fn
    lexical_probe = (lambda candidate: candidate.lstat()) if lstat_fn is None else lstat_fn
    observations: list[ArchiveObservation] = []
    seen: set[Path] = set()
    for supplied in paths:
        try:
            expanded = Path(supplied).expanduser()
            lexical = Path(os.path.abspath(os.fspath(expanded)))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ResourceProbeError(
                f"Cannot resolve archive path {supplied!r}: {error}."
            ) from error
        try:
            lexical_first_result = lexical_probe(lexical)
        except FileNotFoundError as error:
            raise ResourceProbeError(f"Archive does not exist: {lexical}.") from error
        except OSError as error:
            raise ResourceProbeError(f"Cannot inspect archive {lexical}: {error}.") from error
        try:
            lexical_mode = lexical_first_result.st_mode
        except AttributeError as error:
            raise ResourceProbeError(
                f"Cannot safely inspect archive {lexical}: lstat result is incomplete."
            ) from error
        if stat.S_ISLNK(lexical_mode):
            raise ResourceProbeError(
                f"Archive symlink {lexical} is not allowed because adjacent partial "
                "files cannot be accounted for safely."
            )
        lexical_first = _snapshot_fields(
            lexical, lexical_first_result, "archive"
        )
        try:
            resolved = lexical.resolve(strict=False)
            lexical_second_result = lexical_probe(lexical)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            raise ResourceProbeError(
                f"Archive {lexical} changed during inspection: {error}."
            ) from error
        try:
            lexical_second_mode = lexical_second_result.st_mode
        except AttributeError as error:
            raise ResourceProbeError(
                f"Archive {lexical} changed during inspection: lstat result is incomplete."
            ) from error
        if stat.S_ISLNK(lexical_second_mode):
            raise ResourceProbeError(
                f"Archive {lexical} changed during inspection: it became a symlink."
            )
        lexical_second = _snapshot_fields(
            lexical, lexical_second_result, "archive"
        )
        if lexical_first != lexical_second:
            raise ResourceProbeError(
                f"Archive {lexical} changed during inspection; retry once it is stable."
            )
        if resolved in seen:
            raise ResourceProbeError(f"Duplicate archive path after resolution: {resolved}.")
        seen.add(resolved)
        size = _observe_regular_file(
            resolved,
            kind="archive",
            stat_fn=probe,
            allow_missing=False,
        )
        assert size is not None

        partial = Path(f"{resolved}.partial").resolve(strict=False)
        partial_size = _observe_regular_file(
            partial,
            kind="archive partial download",
            stat_fn=probe,
            allow_missing=True,
        )
        observations.append(
            ArchiveObservation(
                path=str(resolved),
                size_bytes=size,
                partial_path=str(partial) if partial_size is not None else None,
                partial_bytes=0 if partial_size is None else partial_size,
            )
        )
    return ArchiveInspection(tuple(observations))


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    request: ResourceRequest
    inspection: ArchiveInspection
    estimate_sources: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "request": self.request.as_dict(),
            "archive_inspection": self.inspection.as_dict(),
            "estimate_sources": dict(self.estimate_sources),
        }


def estimate_resources(
    inspection: ArchiveInspection,
    *,
    final_download_bytes: int,
    planned_partial_bytes: int = 0,
    extracted_bytes: int,
    shards_bytes: int,
    evaluation_bytes: int,
    temporary_bytes: int,
    reserve_bytes: int,
) -> ResourceEstimate:
    """Build a conservative request from explicit estimates and observed partial files."""

    if not isinstance(inspection, ArchiveInspection):
        raise TypeError("inspection must be an ArchiveInspection.")
    supplied = {
        "final_download_bytes": final_download_bytes,
        "planned_partial_bytes": planned_partial_bytes,
        "extracted_bytes": extracted_bytes,
        "shards_bytes": shards_bytes,
        "evaluation_bytes": evaluation_bytes,
        "temporary_bytes": temporary_bytes,
        "reserve_bytes": reserve_bytes,
    }
    checked = {name: _nonnegative_bytes(value, name) for name, value in supplied.items()}
    observed = inspection.total_archive_bytes
    if checked["final_download_bytes"] < observed:
        raise ValueError(
            "The final download estimate cannot be smaller than files already observed: "
            f"final_download_bytes={checked['final_download_bytes']}, observed={observed}."
        )
    remaining_planned_partial = (
        inspection.total_partial_bytes
        if checked["planned_partial_bytes"] == 0
        else max(
            0,
            checked["planned_partial_bytes"] - inspection.total_partial_bytes,
        )
    )
    request = ResourceRequest(
        download=checked["final_download_bytes"],
        extracted=checked["extracted_bytes"],
        shards=checked["shards_bytes"],
        temporary=checked["temporary_bytes"],
        reserve=checked["reserve_bytes"],
        partial=remaining_planned_partial,
        evaluation=checked["evaluation_bytes"],
    )
    sources = (
        ("final_downloads", "caller_supplied"),
        ("extracted_data", "caller_supplied"),
        ("parquet_shards", "caller_supplied"),
        ("evaluation_artifacts", "caller_supplied"),
        ("temporary_workspace", "caller_supplied"),
        ("reserve", "caller_supplied"),
        (
            "partial_downloads",
            "observed_local_files"
            if checked["planned_partial_bytes"] == 0
            else "planned_peak_minus_observed_local_files",
        ),
    )
    return ResourceEstimate(request=request, inspection=inspection, estimate_sources=sources)


@dataclass(frozen=True, slots=True)
class CPUFacts:
    logical_count: int
    used_fallback: bool

    def __post_init__(self) -> None:
        if isinstance(self.logical_count, bool) or not isinstance(self.logical_count, int):
            raise TypeError("logical_count must be a positive integer.")
        if self.logical_count <= 0:
            raise ValueError("logical_count must be positive.")
        if not isinstance(self.used_fallback, bool):
            raise TypeError("used_fallback must be bool.")

    def as_dict(self) -> dict[str, object]:
        return {
            "logical_count": self.logical_count,
            "used_fallback": self.used_fallback,
        }


def discover_cpu(cpu_count_fn: Callable[[], object] | None = None) -> CPUFacts:
    probe = os.cpu_count if cpu_count_fn is None else cpu_count_fn
    try:
        count = probe()
    except Exception:
        return CPUFacts(logical_count=1, used_fallback=True)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        return CPUFacts(logical_count=1, used_fallback=True)
    return CPUFacts(logical_count=count, used_fallback=False)


@dataclass(frozen=True, slots=True)
class RAMFacts:
    total_bytes: int
    available_bytes: int
    platform: str

    def __post_init__(self) -> None:
        total = _positive_integer(self.total_bytes, "RAM total_bytes")
        available = _positive_integer(self.available_bytes, "RAM available_bytes")
        if available > total:
            raise ResourceProbeError(
                f"RAM available_bytes={available} exceeds total_bytes={total}."
            )
        if self.platform not in {"windows", "posix"}:
            raise ResourceProbeError(f"Unknown RAM fact platform {self.platform!r}.")

    def as_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
        }


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def probe_windows_ram(
    *,
    global_memory_status_ex: Callable[[Any], object] | None = None,
    get_last_error: Callable[[], int] | None = None,
) -> RAMFacts:
    """Read physical RAM through the documented Windows GlobalMemoryStatusEx API."""

    if global_memory_status_ex is None:
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            function = kernel32.GlobalMemoryStatusEx
        except (AttributeError, OSError) as error:
            raise ResourceProbeError(
                f"Cannot load Windows GlobalMemoryStatusEx: {error}."
            ) from error
    else:
        function = global_memory_status_ex
    try:
        function.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]  # type: ignore[attr-defined]
        function.restype = ctypes.c_int  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise ResourceProbeError(
            f"Cannot configure Windows GlobalMemoryStatusEx signature: {error}."
        ) from error

    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    try:
        succeeded = function(ctypes.byref(status))
    except Exception as error:
        raise ResourceProbeError(f"GlobalMemoryStatusEx call failed: {error}.") from error
    if not succeeded:
        if get_last_error is None:
            last_error_probe = getattr(ctypes, "get_last_error", lambda: 0)
        else:
            last_error_probe = get_last_error
        try:
            error_code = last_error_probe()
        except Exception:
            error_code = 0
        raise ResourceProbeError(
            f"GlobalMemoryStatusEx failed with Windows error={error_code}."
        )
    return RAMFacts(
        total_bytes=int(status.ullTotalPhys),
        available_bytes=int(status.ullAvailPhys),
        platform="windows",
    )


def probe_posix_ram(*, sysconf_fn: Callable[[str], object] | None = None) -> RAMFacts:
    """Read POSIX page counts without adding a platform dependency."""

    probe = os.sysconf if sysconf_fn is None else sysconf_fn
    try:
        try:
            page_size_raw = probe("SC_PAGE_SIZE")
        except (OSError, ValueError):
            page_size_raw = probe("SC_PAGESIZE")
        physical_pages_raw = probe("SC_PHYS_PAGES")
        available_pages_raw = probe("SC_AVPHYS_PAGES")
    except Exception as error:
        raise ResourceProbeError(f"POSIX RAM probe failed: {error}.") from error
    page_size = _positive_integer(page_size_raw, "SC_PAGE_SIZE")
    physical_pages = _positive_integer(physical_pages_raw, "SC_PHYS_PAGES")
    available_pages = _positive_integer(available_pages_raw, "SC_AVPHYS_PAGES")
    return RAMFacts(
        total_bytes=page_size * physical_pages,
        available_bytes=page_size * available_pages,
        platform="posix",
    )


def discover_ram(
    *,
    platform_name: str | None = None,
    global_memory_status_ex: Callable[[Any], object] | None = None,
    get_last_error: Callable[[], int] | None = None,
    sysconf_fn: Callable[[str], object] | None = None,
) -> RAMFacts:
    selected = os.name if platform_name is None else platform_name
    if selected in {"nt", "windows"}:
        return probe_windows_ram(
            global_memory_status_ex=global_memory_status_ex,
            get_last_error=get_last_error,
        )
    if selected == "posix":
        return probe_posix_ram(sysconf_fn=sysconf_fn)
    raise ResourceProbeError(
        f"Unsupported platform for RAM discovery: {selected!r}; capacity was not guessed."
    )


@dataclass(frozen=True, slots=True)
class DriverInfo:
    nvidia: bool
    driver_version: str | None = None
    device_name: str | None = None
    memory_bytes: int | None = None
    cuda_profile: str | None = None
    device_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.nvidia, bool):
            raise TypeError("nvidia must be bool.")
        for value, name in (
            (self.driver_version, "driver_version"),
            (self.device_name, "device_name"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TypeError(f"{name} must be a non-empty string or None.")
        if self.memory_bytes is not None:
            memory = _nonnegative_bytes(self.memory_bytes, "memory_bytes")
            if memory == 0:
                raise ValueError("memory_bytes must be positive when provided.")
        if self.cuda_profile not in {None, "cu118", "cu124"}:
            raise ValueError("cuda_profile must be cu118, cu124, or None.")
        _device_index(self.device_index)

    def as_dict(self) -> dict[str, object]:
        return {
            "nvidia": self.nvidia,
            "driver_version": self.driver_version,
            "device_name": self.device_name,
            "memory_bytes": self.memory_bytes,
            "cuda_profile": self.cuda_profile,
            "device_index": self.device_index,
        }


NVIDIA_SMI_COMMAND = (
    "nvidia-smi",
    "--query-gpu=driver_version,name,memory.total",
    "--format=csv,noheader",
)

# A driver probe must never delay bootstrap indefinitely. Callers may lower this
# bound for their environment while retaining deterministic timeout handling.
NVIDIA_SMI_TIMEOUT_SECONDS = 10


_NVIDIA_MEMORY = re.compile(r"(?P<mib>[0-9]+)\s+MiB", re.IGNORECASE)


def probe_nvidia_driver(
    *,
    run: Callable[..., Any] | None = None,
    device_index: int | None = None,
    timeout_seconds: float = NVIDIA_SMI_TIMEOUT_SECONDS,
) -> DriverInfo:
    """Probe one NVIDIA GPU with bounded, strict UTF-8 subprocess handling."""

    runner = subprocess.run if run is None else run
    selected_index = _device_index(device_index)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError(
            f"timeout_seconds must be a finite positive number, got {timeout_seconds!r}."
        )
    command = list(NVIDIA_SMI_COMMAND)
    if selected_index is not None:
        command.insert(1, f"--id={selected_index}")
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return DriverInfo(nvidia=False)
    except subprocess.TimeoutExpired as error:
        raise NvidiaProbeError(
            f"nvidia-smi timed out after {timeout_seconds} seconds; "
            "verify the NVIDIA driver is responsive."
        ) from error
    except UnicodeError as error:
        raise NvidiaProbeError(
            f"Could not decode nvidia-smi output as strict UTF-8: {error}."
        ) from error
    except OSError as error:
        raise NvidiaProbeError(f"Could not execute nvidia-smi: {error}.") from error
    try:
        return_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except AttributeError as error:
        raise NvidiaProbeError("nvidia-smi probe returned a malformed process result.") from error
    if isinstance(return_code, bool) or not isinstance(return_code, int):
        raise NvidiaProbeError("nvidia-smi probe returned a malformed exit code.")
    if return_code != 0:
        detail = stderr.strip() if isinstance(stderr, str) and stderr.strip() else "no stderr"
        raise NvidiaProbeError(f"nvidia-smi failed with exit code {return_code}: {detail}.")
    if not isinstance(stdout, str):
        raise NvidiaProbeError("nvidia-smi output is malformed: stdout is not text.")
    rows = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(rows) > 1:
        raise NvidiaProbeError(
            f"nvidia-smi returned multiple GPU rows ({len(rows)}); "
            "explicit device selection is required."
        )
    if len(rows) != 1:
        raise NvidiaProbeError("nvidia-smi output is malformed: expected exactly one GPU row.")
    columns = [column.strip() for column in rows[0].split(",")]
    if len(columns) != 3 or not all(columns):
        raise NvidiaProbeError(
            "nvidia-smi output is malformed: expected driver_version,name,memory.total."
        )
    driver_version, device_name, memory_text = columns
    match = _NVIDIA_MEMORY.fullmatch(memory_text)
    if match is None:
        raise NvidiaProbeError(
            f"nvidia-smi output is malformed: invalid integer memory.total {memory_text!r}."
        )
    memory_mib = int(match.group("mib"))
    if memory_mib <= 0:
        raise NvidiaProbeError(
            f"nvidia-smi output is malformed: invalid memory.total {memory_mib} MiB."
        )
    return DriverInfo(
        nvidia=True,
        driver_version=driver_version,
        device_name=device_name,
        memory_bytes=memory_mib * 1024**2,
        device_index=selected_index,
    )


_CUDA_BUILD_PROFILES = {
    (11, 8): "cu118",
    (12, 4): "cu124",
}


def _profile_for_torch_cuda_build(torch_cuda_version: object) -> str:
    if not isinstance(torch_cuda_version, str) or not torch_cuda_version.strip():
        raise RuntimeError(
            "CUDA profile verification failed: Torch CUDA build version is missing; "
            "no CPU fallback was applied."
        )
    match = re.fullmatch(
        r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)(?:\.[0-9]+)?",
        torch_cuda_version.strip(),
    )
    if match is None:
        raise RuntimeError(
            "CUDA profile verification failed: malformed Torch CUDA build version "
            f"{torch_cuda_version!r}; no CPU fallback was applied."
        )
    build = (int(match.group("major")), int(match.group("minor")))
    try:
        return _CUDA_BUILD_PROFILES[build]
    except KeyError as error:
        normalized = f"{build[0]}.{build[1]}"
        raise RuntimeError(
            "CUDA profile verification failed: unsupported Torch CUDA build "
            f"{normalized}; expected 11.8 or 12.4 and no CPU fallback was applied."
        ) from error


def choose_compute(
    requested: str,
    *,
    driver: DriverInfo,
    torch_cuda: bool,
    torch_cuda_version: str | None = None,
) -> str:
    """Select compute without downgrading a detected CUDA failure.

    CUDA selections resolve only to the pinned ``cu118`` or ``cu124`` profile
    matching the installed Torch CUDA build.
    """

    if requested not in {"auto", "cpu", "cu118", "cu124"}:
        raise ValueError(
            f"Unsupported requested compute profile {requested!r}; use auto, cpu, cu118, or cu124."
        )
    if not isinstance(driver, DriverInfo):
        raise TypeError("driver must be DriverInfo.")
    if not isinstance(torch_cuda, bool):
        raise TypeError("torch_cuda must be bool.")
    if requested == "cpu":
        return "cpu"
    if requested == "auto" and not driver.nvidia:
        return "cpu"
    if not driver.nvidia:
        raise RuntimeError(
            "NVIDIA driver/GPU is required for requested compute profile "
            f"{requested!r}; no fallback was applied."
        )
    if not torch_cuda:
        raise RuntimeError(
            "CUDA profile verification failed: an NVIDIA driver/GPU was detected but "
            "Torch CUDA is unavailable; no CPU fallback was applied."
        )
    installed_profile = _profile_for_torch_cuda_build(torch_cuda_version)
    if requested == "auto":
        return installed_profile
    if requested != installed_profile:
        raise RuntimeError(
            "CUDA profile verification failed: requested profile "
            f"{requested} does not match Torch CUDA build {torch_cuda_version!r} "
            f"({installed_profile}); no CPU fallback was applied."
        )
    return requested


@dataclass(frozen=True, slots=True)
class TorchVerification:
    selected_profile: str
    device: str
    device_name: str
    torch_version: str
    torch_cuda_version: str | None
    device_index: int | None = None

    def __post_init__(self) -> None:
        if self.selected_profile not in {"cpu", "cu118", "cu124"}:
            raise ValueError(f"Unknown selected_profile {self.selected_profile!r}.")
        for value, name in (
            (self.device, "device"),
            (self.device_name, "device_name"),
            (self.torch_version, "torch_version"),
        ):
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string.")
        if self.torch_cuda_version is not None and (
            not isinstance(self.torch_cuda_version, str) or not self.torch_cuda_version
        ):
            raise TypeError("torch_cuda_version must be a non-empty string or None.")
        _device_index(self.device_index)

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_profile": self.selected_profile,
            "device": self.device,
            "device_name": self.device_name,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "device_index": self.device_index,
        }


def verify_torch_compute(
    selected_profile: str,
    *,
    torch_module: Any | None = None,
    importer: Callable[[str], Any] | None = None,
    device_index: int | None = None,
    driver: DriverInfo | None = None,
) -> TorchVerification:
    """Lazily smoke-test one explicit runtime device.

    Unlike the installer check, an explicit CPU runtime choice is valid even when
    the installed Torch wheel includes CUDA; only CPU tensor work is performed.
    """

    if selected_profile not in {"cpu", "cu118", "cu124"}:
        raise ValueError(f"Unknown selected compute profile {selected_profile!r}.")
    requested_index = _device_index(device_index)
    if driver is not None and not isinstance(driver, DriverInfo):
        raise TypeError("driver must be DriverInfo or None.")
    cuda_selected = selected_profile != "cpu"
    if not cuda_selected:
        if requested_index is not None:
            raise ValueError("device_index must be None for CPU verification.")
        selected_index = None
    else:
        selected_index = 0 if requested_index is None else requested_index
        if driver is not None:
            if not driver.nvidia:
                raise RuntimeError(
                    "CUDA device index verification failed: DriverInfo reports no "
                    "NVIDIA device; no CPU fallback was applied."
                )
            if (
                driver.device_index is not None
                and driver.device_index != selected_index
            ):
                raise RuntimeError(
                    "CUDA device index mismatch: DriverInfo selected "
                    f"cuda:{driver.device_index} but verification requested "
                    f"cuda:{selected_index}; no CPU fallback was applied."
                )
    if torch_module is None:
        import_module = importlib.import_module if importer is None else importer
        try:
            torch_module = import_module("torch")
        except Exception as error:
            raise RuntimeError(f"Torch import failed: {error}.") from error
    try:
        torch_version = torch_module.__version__
    except Exception as error:
        raise RuntimeError(f"Torch version verification failed: {error}.") from error
    if not isinstance(torch_version, str) or not torch_version:
        raise RuntimeError(
            f"Torch version verification failed: invalid __version__ {torch_version!r}."
        )

    try:
        torch_cuda_version = torch_module.version.cuda
    except Exception as error:
        raise RuntimeError(f"Torch CUDA build version verification failed: {error}.") from error
    if torch_cuda_version is not None and (
        not isinstance(torch_cuda_version, str) or not torch_cuda_version
    ):
        raise RuntimeError(
            "Torch CUDA build version verification failed: "
            f"invalid version {torch_cuda_version!r}."
        )

    if cuda_selected:
        try:
            cuda_available = torch_module.cuda.is_available()
        except Exception as error:
            raise RuntimeError(f"CUDA profile verification failed: {error}.") from error
        if cuda_available is not True:
            raise RuntimeError(
                "CUDA profile verification failed: Torch reports CUDA unavailable; "
                "no CPU fallback was applied."
            )
        if torch_cuda_version is None:
            raise RuntimeError(
                "CUDA profile verification failed: Torch does not report a CUDA build version; "
                "no CPU fallback was applied."
            )
        installed_profile = _profile_for_torch_cuda_build(torch_cuda_version)
        if selected_profile != installed_profile:
            raise RuntimeError(
                "CUDA profile verification failed: selected profile "
                f"{selected_profile} does not match Torch CUDA build "
                f"{torch_cuda_version!r} ({installed_profile}); "
                "no CPU fallback was applied."
            )
        assert selected_index is not None
        device = f"cuda:{selected_index}"
    else:
        device = "cpu"

    try:
        tensor = torch_module.ones((1,), device=device)
    except Exception as error:
        raise RuntimeError(
            f"Torch tensor allocation verification failed on {device}: {error}."
        ) from error
    try:
        actual = (tensor + tensor).item()
    except Exception as error:
        raise RuntimeError(
            f"Torch tensor operation verification failed on {device}: {error}."
        ) from error
    if actual != 2:
        raise RuntimeError(
            f"Torch tensor operation verification failed on {device}: "
            f"expected=2, actual={actual!r}."
        )

    if cuda_selected:
        assert selected_index is not None
        try:
            device_name = torch_module.cuda.get_device_name(selected_index)
        except Exception as error:
            raise RuntimeError(f"CUDA device-name verification failed: {error}.") from error
        if not isinstance(device_name, str) or not device_name:
            raise RuntimeError(
                f"CUDA device-name verification failed: invalid name {device_name!r}."
            )
        try:
            torch_module.cuda.synchronize(selected_index)
        except Exception as error:
            raise RuntimeError(f"CUDA synchronization verification failed: {error}.") from error
    else:
        device_name = "CPU"
    return TorchVerification(
        selected_profile=selected_profile,
        device=device,
        device_name=device_name,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        device_index=selected_index,
    )


DEFAULT_PACKAGE_NAMES = (
    "torch",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
)


def collect_package_versions(
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
    *,
    version_fn: Callable[[str], str] | None = None,
) -> dict[str, str | None]:
    """Read installed versions through metadata without importing scientific packages."""

    lookup = importlib.metadata.version if version_fn is None else version_fn
    versions: dict[str, str | None] = {}
    for package in package_names:
        if not isinstance(package, str) or not package:
            raise TypeError("package names must be non-empty strings.")
        if package in versions:
            raise ValueError(f"Duplicate package name {package!r}.")
        try:
            version = lookup(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            continue
        except Exception as error:
            raise ResourceProbeError(
                f"Could not inspect installed package version for {package}: {error}."
            ) from error
        if not isinstance(version, str) or not version:
            raise ResourceProbeError(
                f"Installed package version for {package} is malformed: {version!r}."
            )
        versions[package] = version
    return versions


@dataclass(frozen=True, slots=True)
class PostInstallReport:
    verification: TorchVerification
    driver: DriverInfo
    cpu: CPUFacts
    ram: RAMFacts
    disk: DiskFacts
    installed_packages: tuple[tuple[str, str | None], ...]

    def as_dict(self) -> dict[str, object]:
        compute = self.verification.as_dict()
        compute["driver"] = self.driver.as_dict()
        return {
            "compute": compute,
            "resources": {
                "cpu": self.cpu.as_dict(),
                "ram": self.ram.as_dict(),
                "disk": self.disk.as_dict(),
            },
            "installed_packages": dict(self.installed_packages),
        }


def build_post_install_report(
    *,
    verification: TorchVerification,
    driver: DriverInfo,
    cpu: CPUFacts,
    ram: RAMFacts,
    disk: DiskFacts,
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
    version_fn: Callable[[str], str] | None = None,
) -> PostInstallReport:
    """Aggregate mutually consistent compute and read-only resource facts.

    A CPU runtime choice may coexist with an NVIDIA driver and CUDA-enabled Torch
    wheel. CUDA selections, however, must agree on profile, device, and index.
    """

    if not isinstance(verification, TorchVerification):
        raise TypeError("verification must be TorchVerification.")
    if not isinstance(driver, DriverInfo):
        raise TypeError("driver must be DriverInfo.")
    if not isinstance(cpu, CPUFacts):
        raise TypeError("cpu must be CPUFacts.")
    if not isinstance(ram, RAMFacts):
        raise TypeError("ram must be RAMFacts.")
    if not isinstance(disk, DiskFacts):
        raise TypeError("disk must be DiskFacts.")
    cuda_selected = verification.selected_profile != "cpu"
    if not cuda_selected:
        if verification.device != "cpu" or verification.device_index is not None:
            raise ResourceProbeError(
                "Post-install report facts are contradictory: CPU verification "
                "must use device='cpu' with no device index."
            )
    else:
        if not driver.nvidia:
            raise ResourceProbeError(
                "Post-install report facts are contradictory: CUDA was verified "
                "but DriverInfo reports no NVIDIA device."
            )
        if verification.device_index is None:
            raise ResourceProbeError(
                "Post-install report facts are contradictory: CUDA verification "
                "has no selected device index."
            )
        expected_device = f"cuda:{verification.device_index}"
        if verification.device != expected_device:
            raise ResourceProbeError(
                "Post-install report facts are contradictory: CUDA device "
                f"{verification.device!r} does not match index "
                f"{verification.device_index}."
            )
        if (
            driver.device_index is not None
            and driver.device_index != verification.device_index
        ):
            raise ResourceProbeError(
                "Post-install report facts are contradictory: DriverInfo selected "
                f"cuda:{driver.device_index} but Torch verified {expected_device}."
            )
        if (
            driver.cuda_profile is not None
            and driver.cuda_profile != verification.selected_profile
        ):
            raise ResourceProbeError(
                "Post-install report facts are contradictory: DriverInfo profile "
                f"{driver.cuda_profile} does not match verified profile "
                f"{verification.selected_profile}."
            )
        try:
            installed_profile = _profile_for_torch_cuda_build(
                verification.torch_cuda_version
            )
        except RuntimeError as error:
            raise ResourceProbeError(
                "Post-install report facts are contradictory: verified CUDA build "
                f"is invalid: {error}."
            ) from error
        if installed_profile != verification.selected_profile:
            raise ResourceProbeError(
                "Post-install report facts are contradictory: verified profile "
                f"{verification.selected_profile} does not match Torch CUDA build "
                f"{verification.torch_cuda_version!r}."
            )
    packages = collect_package_versions(package_names, version_fn=version_fn)
    return PostInstallReport(
        verification=verification,
        driver=driver,
        cpu=cpu,
        ram=ram,
        disk=disk,
        installed_packages=tuple(packages.items()),
    )
