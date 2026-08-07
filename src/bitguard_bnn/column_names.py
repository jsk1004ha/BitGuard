"""Shared CSV column-name normalization and ambiguity checks."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable


def normalize_csv_column_name(value: object) -> str:
    """Return the canonical spelling used at CSV inspection and read boundaries."""

    return unicodedata.normalize("NFC", str(value).strip())


def csv_column_name_key(value: object) -> str:
    """Return the comparison key shared by schema and training readers."""

    return normalize_csv_column_name(value).casefold()


def normalize_csv_column_names(
    columns: Iterable[object],
    *,
    source: object,
) -> tuple[str, ...]:
    """Normalize CSV columns while rejecting control characters and ambiguity."""

    normalized: list[str] = []
    seen: set[str] = set()
    for value in columns:
        raw = str(value)
        if any(ord(character) < 32 or ord(character) == 127 for character in raw):
            raise ValueError(
                f"CSV header contains control characters in {source}: {raw!r}"
            )
        column = normalize_csv_column_name(raw)
        if not column:
            raise ValueError(f"CSV header contains an empty column: {source}")
        key = csv_column_name_key(column)
        if key in seen:
            raise ValueError(
                "CSV header has a duplicate column after trimming, NFC "
                f"normalization, and case folding in {source}: {raw!r}"
            )
        seen.add(key)
        normalized.append(column)
    if not normalized:
        raise ValueError(f"empty CSV header: {source}")
    return tuple(normalized)
