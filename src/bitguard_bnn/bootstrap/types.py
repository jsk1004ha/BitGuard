from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field)


def _string_tuple(value: object, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    project_url: str
    download_url: str | None
    doi: str | None
    license_name: str
    expected_patterns: tuple[str, ...]
    required_columns: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetSpec:
        fields = {
            "name",
            "project_url",
            "download_url",
            "doi",
            "license_name",
            "expected_patterns",
            "required_columns",
        }
        missing = fields - set(value)
        extra = set(value) - fields
        if missing or extra:
            raise ValueError(
                f"invalid dataset registry fields: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return cls(
            name=_required_string(value["name"], "name"),
            project_url=_required_string(value["project_url"], "project_url"),
            download_url=_optional_string(value["download_url"], "download_url"),
            doi=_optional_string(value["doi"], "doi"),
            license_name=_required_string(value["license_name"], "license_name"),
            expected_patterns=_string_tuple(
                value["expected_patterns"], "expected_patterns", allow_empty=False
            ),
            required_columns=_string_tuple(
                value["required_columns"], "required_columns", allow_empty=True
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "project_url": self.project_url,
            "download_url": self.download_url,
            "doi": self.doi,
            "license_name": self.license_name,
            "expected_patterns": list(self.expected_patterns),
            "required_columns": list(self.required_columns),
        }


@dataclass(frozen=True)
class BootstrapOptions:
    datasets: tuple[str, ...]
    botiot_source: Path | None
    data_root: Path
    runs_root: Path
    compute: str
    prepare_only: bool
    install_system_tools: bool
    accepted_botiot_license: bool
    restart_stage: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "datasets": list(self.datasets),
            "botiot_source": str(self.botiot_source) if self.botiot_source is not None else None,
            "data_root": str(self.data_root),
            "runs_root": str(self.runs_root),
            "compute": self.compute,
            "prepare_only": self.prepare_only,
            "install_system_tools": self.install_system_tools,
            "accepted_botiot_license": self.accepted_botiot_license,
            "restart_stage": self.restart_stage,
        }
