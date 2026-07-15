"""Read-only reporting for bootstrap artifacts retained after safe failures."""

from __future__ import annotations

import os
import platform
import re
import shlex
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Any


_ATOMIC_JSON_TEMP = re.compile(r".+\.json\.[0-9a-f]{32}\.tmp\Z", re.IGNORECASE)


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def inspection_command(paths: Iterable[str]) -> str:
    supplied = tuple(paths)
    warning = (
        "Do not delete automatically. Inspect identity, link type, and size before "
        "any manual cleanup."
    )
    if platform.system().casefold().startswith("win"):
        if not supplied:
            return (
                f"Write-Output "
                f"{_powershell_literal(warning + ' No retained artifacts found.')}"
            )
        literals = ", ".join(_powershell_literal(path) for path in supplied)
        return (
            f"$paths = @({literals}); Write-Output {_powershell_literal(warning)}; "
            "Get-Item -Force -LiteralPath $paths | "
            "Select-Object FullName,Attributes,LinkType,Length"
        )
    quoted_warning = shlex.quote(warning)
    if not supplied:
        return f"printf '%s\\n' {quoted_warning}"
    quoted_paths = " ".join(shlex.quote(path) for path in supplied)
    return f"printf '%s\\n' {quoted_warning} && ls -ld -- {quoted_paths}"


def _unsafe_link_or_reparse(result: os.stat_result) -> bool:
    return stat.S_ISLNK(result.st_mode) or bool(getattr(result, "st_reparse_tag", 0))


def _debt_kind(name: str) -> str | None:
    if name.startswith(".bitguard-retired-"):
        return "retired"
    if name.startswith(".bitguard-acquire-"):
        return "acquisition_staging"
    if name.startswith(".bitguard-extract-"):
        return "extraction_staging"
    if name.casefold().endswith(".partial"):
        return "partial"
    if _ATOMIC_JSON_TEMP.fullmatch(name):
        return "atomic_json_temporary"
    return None


def _scan_error(path: Path, message: str) -> dict[str, str]:
    return {"path": str(path), "error": message}


def _artifact_sizes(
    path: Path,
    *,
    globally_seen: set[tuple[int, int]],
    scan_errors: list[dict[str, str]],
) -> tuple[int, int]:
    apparent = 0
    unique = 0

    def visit(candidate: Path) -> None:
        nonlocal apparent, unique
        try:
            current = candidate.lstat()
        except OSError as error:
            scan_errors.append(_scan_error(candidate, f"lstat failed: {error}"))
            return
        if _unsafe_link_or_reparse(current):
            scan_errors.append(
                _scan_error(candidate, "link or reparse point was not traversed")
            )
            return
        if stat.S_ISREG(current.st_mode):
            size = max(0, int(current.st_size))
            apparent += size
            identity = (int(current.st_dev), int(current.st_ino))
            if identity not in globally_seen:
                globally_seen.add(identity)
                unique += size
            return
        if not stat.S_ISDIR(current.st_mode):
            scan_errors.append(
                _scan_error(candidate, "special filesystem entry was not traversed")
            )
            return
        try:
            with os.scandir(candidate) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            scan_errors.append(_scan_error(candidate, f"scan failed: {error}"))
            return
        for entry in entries:
            child = Path(entry.path)
            visit(child)

    visit(path)
    return apparent, unique


def scan_cleanup_debt(roots: Iterable[Path | str]) -> dict[str, Any]:
    """Describe retained quarantine/staging paths without deleting anything."""

    artifacts: list[dict[str, object]] = []
    apparent_total = 0
    unique_total = 0
    globally_seen: set[tuple[int, int]] = set()
    seen_paths: set[Path] = set()
    scan_errors: list[dict[str, str]] = []
    for supplied in roots:
        root = Path(os.path.abspath(Path(supplied).expanduser()))
        if root in seen_paths:
            continue
        seen_paths.add(root)
        try:
            root_result = root.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            scan_errors.append(_scan_error(root, f"lstat failed: {error}"))
            continue
        if _unsafe_link_or_reparse(root_result) or not stat.S_ISDIR(
            root_result.st_mode
        ):
            scan_errors.append(
                _scan_error(root, "cleanup root is not a safe ordinary directory")
            )
            continue
        try:
            with os.scandir(root) as iterator:
                children = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            scan_errors.append(_scan_error(root, f"scan failed: {error}"))
            continue
        if root.name == "preparation-work":
            generation_paths: list[Path] = []
            for dataset_entry in children:
                dataset_path = Path(dataset_entry.path)
                try:
                    dataset_stat = dataset_path.lstat()
                except OSError as error:
                    scan_errors.append(
                        _scan_error(dataset_path, f"lstat failed: {error}")
                    )
                    continue
                if _unsafe_link_or_reparse(dataset_stat):
                    scan_errors.append(
                        _scan_error(
                            dataset_path,
                            "preparation-work link or reparse point was not traversed",
                        )
                    )
                    continue
                if not stat.S_ISDIR(dataset_stat.st_mode):
                    scan_errors.append(
                        _scan_error(
                            dataset_path,
                            "preparation-work dataset entry is not a directory",
                        )
                    )
                    continue
                try:
                    with os.scandir(dataset_path) as iterator:
                        generation_paths.extend(
                            Path(entry.path)
                            for entry in sorted(iterator, key=lambda value: value.name)
                        )
                except OSError as error:
                    scan_errors.append(
                        _scan_error(dataset_path, f"scan failed: {error}")
                    )
            for candidate in generation_paths:
                apparent, unique = _artifact_sizes(
                    candidate,
                    globally_seen=globally_seen,
                    scan_errors=scan_errors,
                )
                apparent_total += apparent
                unique_total += unique
                artifacts.append(
                    {
                        "path": str(candidate),
                        "kind": "preparation_work_generation",
                        "apparent_bytes": apparent,
                        "unique_bytes": unique,
                    }
                )
            continue
        for entry in children:
            kind = _debt_kind(entry.name)
            if kind is None:
                continue
            candidate = Path(entry.path)
            apparent, unique = _artifact_sizes(
                candidate,
                globally_seen=globally_seen,
                scan_errors=scan_errors,
            )
            apparent_total += apparent
            unique_total += unique
            artifacts.append(
                {
                    "path": str(candidate),
                    "kind": kind,
                    "apparent_bytes": apparent,
                    "unique_bytes": unique,
                }
            )
    return {
        "artifacts": artifacts,
        "apparent_bytes": apparent_total,
        "unique_bytes": unique_total,
        "scan_errors": scan_errors,
        "recovery_command": inspection_command(
            str(artifact["path"]) for artifact in artifacts
        ),
    }
