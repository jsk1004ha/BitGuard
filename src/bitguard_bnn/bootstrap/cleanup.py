"""Read-only reporting for bootstrap artifacts retained after safe failures."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Any


_DEBT_PREFIXES = (".bitguard-retired-", ".bitguard-extract-")


def _artifact_sizes(path: Path) -> tuple[int, int]:
    apparent = 0
    unique = 0
    seen: set[tuple[int, int]] = set()
    candidates = (path,) if not path.is_dir() else path.rglob("*")
    for candidate in candidates:
        try:
            result = candidate.lstat()
        except OSError:
            continue
        if not stat.S_ISREG(result.st_mode):
            continue
        size = max(0, int(result.st_size))
        apparent += size
        identity = (int(result.st_dev), int(result.st_ino))
        if identity not in seen:
            seen.add(identity)
            unique += size
    return apparent, unique


def scan_cleanup_debt(roots: Iterable[Path | str]) -> dict[str, Any]:
    """Describe retained quarantine/staging paths without deleting anything."""

    artifacts: list[dict[str, object]] = []
    apparent_total = 0
    unique_total = 0
    globally_seen: set[tuple[int, int]] = set()
    seen_paths: set[Path] = set()
    for supplied in roots:
        root = Path(supplied).expanduser().resolve(strict=False)
        if root in seen_paths or not root.is_dir():
            continue
        seen_paths.add(root)
        try:
            children = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for candidate in children:
            if not candidate.name.startswith(_DEBT_PREFIXES):
                continue
            apparent, _ = _artifact_sizes(candidate)
            unique = 0
            members = (candidate,) if not candidate.is_dir() else candidate.rglob("*")
            for member in members:
                try:
                    result = member.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(result.st_mode):
                    continue
                identity = (int(result.st_dev), int(result.st_ino))
                if identity in globally_seen:
                    continue
                globally_seen.add(identity)
                unique += max(0, int(result.st_size))
            apparent_total += apparent
            unique_total += unique
            artifacts.append(
                {
                    "path": str(candidate),
                    "kind": (
                        "retired"
                        if candidate.name.startswith(".bitguard-retired-")
                        else "extraction_staging"
                    ),
                    "apparent_bytes": apparent,
                    "unique_bytes": unique,
                }
            )
    return {
        "artifacts": artifacts,
        "apparent_bytes": apparent_total,
        "unique_bytes": unique_total,
        "recovery_command": (
            "Do not delete automatically. Inspect each listed path and its link count "
            "first; remove it manually only after confirming no active bootstrap uses it."
        ),
    }
