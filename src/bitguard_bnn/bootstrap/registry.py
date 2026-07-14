from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import DatasetSpec


def load_registry(path: Path | None = None) -> dict[str, DatasetSpec]:
    registry_path = path or Path(__file__).with_name("datasets.json")
    try:
        raw: Any = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load dataset registry {registry_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("dataset registry must be a JSON object")

    registry: dict[str, DatasetSpec] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("dataset registry entries must map string keys to objects")
        spec = DatasetSpec.from_mapping(value)
        if spec.name != key:
            raise ValueError(f"dataset registry key {key!r} does not match name {spec.name!r}")
        registry[key] = spec
    if tuple(registry) != ("nbaiot", "botiot"):
        raise ValueError("dataset registry must contain exactly nbaiot then botiot")
    return registry
