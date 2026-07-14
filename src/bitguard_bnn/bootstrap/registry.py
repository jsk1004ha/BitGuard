from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .types import DatasetSpec

NBAIOT_PROJECT_URL = (
    "https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot"
)
NBAIOT_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/442/"
    "detection%2Bof%2Biot%2Bbotnet%2Battacks%2Bn%2Bbaiot.zip"
)
NBAIOT_DOI = "10.24432/C5RC8J"
BOTIOT_PROJECT_URL = "https://research.unsw.edu.au/projects/bot-iot-dataset"


def _validate_https_url(dataset: str, field: str, value: str | None) -> None:
    if value is None:
        return
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError(f"{dataset}.{field} is not a valid HTTPS URL: {exc}") from exc
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(f"{dataset}.{field} must be an HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{dataset}.{field} must not contain credentials")


def _validate_official_sources(registry: dict[str, DatasetSpec]) -> None:
    for name, spec in registry.items():
        _validate_https_url(name, "project_url", spec.project_url)
        _validate_https_url(name, "download_url", spec.download_url)

    nbaiot = registry["nbaiot"]
    if nbaiot.project_url != NBAIOT_PROJECT_URL:
        raise ValueError("nbaiot.project_url must use the official UCI dataset-442 identity")
    if nbaiot.download_url != NBAIOT_DOWNLOAD_URL:
        raise ValueError("nbaiot.download_url must use the official UCI dataset-442 identity")
    if nbaiot.doi != NBAIOT_DOI:
        raise ValueError(f"nbaiot.doi must be {NBAIOT_DOI}")

    botiot = registry["botiot"]
    if botiot.project_url != BOTIOT_PROJECT_URL:
        raise ValueError("botiot.project_url must use the official UNSW project identity")
    if botiot.download_url is not None:
        raise ValueError("botiot must not define download_url; provide the source explicitly")


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
    _validate_official_sources(registry)
    return registry
