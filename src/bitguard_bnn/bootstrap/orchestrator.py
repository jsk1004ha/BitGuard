"""Idempotent acquisition orchestration through verified CSV sources."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .cleanup import scan_cleanup_debt
from .download import DownloadResult, download_file, sanitize_url
from .extract import ExtractionResult, extract_rar, extract_zip
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
from .state import BootstrapStateStore, BootstrapWriterLock, STAGE_ORDER
from .types import BootstrapOptions, DatasetSpec


REPORT_FORMAT_VERSION = 1
UNKNOWN_REMOTE_ARCHIVE_BYTES = 4 * 1024**3
ARCHIVE_EXPANSION_FACTOR = 12
REPORT_AND_METADATA_BYTES = 64 * 1024**2
DEFAULT_DISK_RESERVE_BYTES = 2 * 1024**3
_URL_IN_TEXT = re.compile(r"https?://[^\s]+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    input_signature: Callable[[], str]
    run: Callable[[], Sequence[Path]]


def _default_compute_resolver(requested: str) -> dict[str, object]:
    driver = probe_nvidia_driver()
    if requested == "cpu":
        selected = "cpu"
    else:
        try:
            import torch
        except Exception as error:  # pragma: no cover - installation failure path
            raise RuntimeError(f"Torch import failed during compute selection: {error}") from error
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
    downloader: Callable[..., DownloadResult] = download_file
    zip_extractor: Callable[..., ExtractionResult] = extract_zip
    rar_extractor: Callable[..., ExtractionResult] = extract_rar
    inspector: Callable[..., SchemaInspectionReport] = inspect_csv_dataset
    compute_resolver: Callable[[str], Mapping[str, object]] = _default_compute_resolver


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
        raise RuntimeError(f"Bootstrap source must be a regular non-symlink file: {path}")
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
    root = path.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Bootstrap source must be a non-symlink directory: {path}")
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
        raise RuntimeError(f"Bootstrap source directory contains no regular files: {root}")
    return _json_signature(records), sum(item[1] for item in records), tuple(files)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("atomic JSON write made no progress")
        offset += written


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)


def _safe_url_or_text(value: object) -> object:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return value
    try:
        if value.lower().startswith(("http://", "https://")):
            return sanitize_url(value)
    except Exception:
        return "<redacted-invalid-url>"
    return value


def _safe_error(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"

    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0).rstrip(".,;:)")
        suffix = match.group(0)[len(candidate) :]
        try:
            return sanitize_url(candidate) + suffix
        except Exception:
            return "<redacted-url>" + suffix

    return _URL_IN_TEXT.sub(replace, text)


def _existing_parent(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists():
        if candidate.parent == candidate:
            raise RuntimeError(f"No existing parent is available for bootstrap path {path}")
        candidate = candidate.parent
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(f"Bootstrap parent must be an existing trusted directory: {candidate}")
    return candidate.resolve(strict=True)


def _lock_path(options: BootstrapOptions) -> Path:
    parent = _existing_parent(options.data_root.parent)
    identity = _json_signature(
        {"data_root": str(options.data_root), "runs_root": str(options.runs_root)}
    )[:20]
    return parent / f".bitguard-bootstrap-{identity}.lock"


def _copy_file(source: Path, destination: Path, expected_digest: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    except BaseException:
        # The private candidate is intentionally retained if publication fails.
        raise
    return destination


def _copy_tree(source: Path, destination: Path, expected_digest: str) -> tuple[Path, ...]:
    if destination.exists():
        actual, _size, files = _tree_digest(destination)
        if actual == expected_digest:
            return files
        raise RuntimeError(
            f"Existing acquisition directory differs from its source: {destination}. "
            "Preserve it for inspection and use a new data root."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    os.rename(staging, destination)
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
    if stage is None:
        return "Inspect the bootstrap report, correct the input or resource error, and rerun the original command."
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
    registry = load_registry()
    metadata_root = options.data_root / ".bitguard"
    report_path = metadata_root / "bootstrap-report.json"
    state_path = metadata_root / "bootstrap-state.json"
    original = {
        "botiot_source": None,
        "data_root": str(options.data_root),
        "runs_root": str(options.runs_root),
        **dict(raw_inputs or {}),
    }
    safe_original = {key: _safe_url_or_text(value) for key, value in original.items()}
    safe_resolved = {
        key: _safe_url_or_text(value)
        for key, value in options.to_dict().items()
    }
    try:
        lock_path = _lock_path(options)
    except BaseException as error:
        fallback = Path.cwd() / (
            f".bitguard-bootstrap-failure-{os.getpid()}-{uuid.uuid4().hex}.json"
        )
        result: dict[str, object] = {
            "version": REPORT_FORMAT_VERSION,
            "status": "failed",
            "last_completed_stage": None,
            "failed_stage": "preflight",
            "error": _safe_error(error),
            "recovery_command": _recovery("preflight"),
            "inputs": {"original": safe_original, "resolved": safe_resolved},
            "report_path": str(fallback),
        }
        _write_json(fallback, result)
        return result
    executed: list[str] = []
    reused: list[str] = []
    failed_stage: str | None = None
    last_completed: str | None = None
    compute: dict[str, object] | None = None
    manifests: dict[str, str] = {}
    schemas: dict[str, str] = {}
    cleanup_roots: set[Path] = {metadata_root, options.data_root}

    def report(status: str, error: BaseException | None = None) -> dict[str, object]:
        debt = scan_cleanup_debt(tuple(sorted(cleanup_roots, key=str)))
        result: dict[str, object] = {
            "version": REPORT_FORMAT_VERSION,
            "status": status,
            "last_completed_stage": last_completed,
            "failed_stage": failed_stage,
            "error": None if error is None else _safe_error(error),
            "recovery_command": _recovery(failed_stage) if error is not None else None,
            "inputs": {"original": safe_original, "resolved": safe_resolved},
            "compute": compute,
            "state": str(state_path),
            "manifests": manifests,
            "schema_reports": schemas,
            "cleanup_debt": debt,
            "executed_stages": executed,
            "reused_stages": reused,
            "threat_model": (
                "Defends untrusted network/archive input and cooperative writers in a "
                "trusted workspace. Malicious same-account parent-namespace or hardlink "
                "mutation is outside this contract."
            ),
        }
        _write_json(report_path, result)
        return result

    current_stage: str | None = None
    try:
        with BootstrapWriterLock(lock_path):
            try:
                current_stage = "preflight"
                source_context: dict[str, dict[str, object]] = {}
                if "nbaiot" in options.datasets:
                    if deps.nbaiot_archive is None:
                        remote_bytes = deps.nbaiot_download_size or UNKNOWN_REMOTE_ARCHIVE_BYTES
                        source_context["nbaiot"] = {
                            "kind": "zip",
                            "digest": None,
                            "bytes": remote_bytes,
                            "source": None,
                        }
                    else:
                        source = deps.nbaiot_archive.expanduser().resolve(strict=True)
                        if source.suffix.casefold() != ".zip":
                            raise RuntimeError("Injected N-BaIoT source must be a ZIP archive")
                        digest, size = _regular_digest(source)
                        source_context["nbaiot"] = {
                            "kind": "zip",
                            "digest": digest,
                            "bytes": size,
                            "source": source,
                        }
                if "botiot" in options.datasets:
                    if not options.accepted_botiot_license or options.botiot_source is None:
                        raise RuntimeError(
                            "BoT-IoT requires a local source and explicit academic-license acknowledgement"
                        )
                    source = options.botiot_source.resolve(strict=True)
                    try:
                        source.relative_to(options.data_root)
                    except ValueError:
                        pass
                    else:
                        raise RuntimeError("BoT-IoT source must be outside the bootstrap data root")
                    try:
                        source.relative_to(options.runs_root)
                    except ValueError:
                        pass
                    else:
                        raise RuntimeError("BoT-IoT source must be outside the bootstrap runs root")
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

                download_bytes = sum(int(item["bytes"]) for item in source_context.values())
                archive_bytes = sum(
                    int(item["bytes"])
                    for item in source_context.values()
                    if item["kind"] in {"zip", "rar"}
                )
                directory_bytes = download_bytes - archive_bytes
                estimate = estimate_resources(
                    ArchiveInspection(()),
                    final_download_bytes=download_bytes,
                    planned_partial_bytes=(
                        int(source_context.get("nbaiot", {}).get("bytes", 0))
                        if deps.nbaiot_archive is None and "nbaiot" in source_context
                        else 0
                    ),
                    extracted_bytes=archive_bytes * ARCHIVE_EXPANSION_FACTOR + directory_bytes,
                    shards_bytes=0,
                    evaluation_bytes=0,
                    temporary_bytes=REPORT_AND_METADATA_BYTES,
                    reserve_bytes=DEFAULT_DISK_RESERVE_BYTES,
                )
                if deps.available_bytes is None:
                    available = shutil.disk_usage(_existing_parent(options.data_root)).free
                else:
                    available = deps.available_bytes
                disk = require_disk(estimate.request, available_bytes=available)

                metadata_root.mkdir(parents=True, exist_ok=True)
                options.runs_root.mkdir(parents=True, exist_ok=True)
                state = BootstrapStateStore(state_path)
                if options.restart_stage is not None:
                    state.invalidate_from(options.restart_stage, STAGE_ORDER)

                preflight_path = metadata_root / "preflight.json"
                environment_path = metadata_root / "environment.json"
                acquisition_root = metadata_root / "acquired"
                extraction_root = options.data_root / "raw"
                manifest_root = metadata_root / "manifests"
                schema_root = metadata_root / "schema"
                cleanup_roots.update(
                    {
                        metadata_root,
                        acquisition_root,
                        extraction_root,
                        manifest_root,
                        schema_root,
                    }
                )

                acquired: dict[str, Path] = {}
                raw_roots: dict[str, Path] = {}
                for dataset, context in source_context.items():
                    digest = context["digest"]
                    token = str(digest or "official")
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

                def run_environment() -> Sequence[Path]:
                    nonlocal compute
                    compute = dict(deps.compute_resolver(options.compute))
                    _write_json(environment_path, compute)
                    return (environment_path,)

                def run_acquire() -> Sequence[Path]:
                    acquisition_root.mkdir(parents=True, exist_ok=True)
                    outputs: list[Path] = []
                    acquisition_report: dict[str, object] = {}
                    for dataset, context in source_context.items():
                        destination = acquired[dataset]
                        source = context["source"]
                        digest = context["digest"]
                        kind = str(context["kind"])
                        if dataset == "nbaiot" and source is None:
                            spec = registry[dataset]
                            assert spec.download_url is not None
                            prior_hash: str | None = None
                            prior = metadata_root / "acquisition.json"
                            if prior.is_file():
                                try:
                                    prior_payload = json.loads(prior.read_text(encoding="utf-8"))
                                    value = prior_payload["datasets"]["nbaiot"]["sha256"]
                                    if isinstance(value, str):
                                        prior_hash = value
                                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                                    prior_hash = None
                            result = deps.downloader(
                                spec.download_url,
                                destination,
                                expected_sha256=prior_hash,
                            )
                            acquisition_report[dataset] = result.to_dict()
                            outputs.append(destination)
                        elif kind == "directory":
                            assert isinstance(source, Path) and isinstance(digest, str)
                            outputs.extend(_copy_tree(source, destination, digest))
                            acquisition_report[dataset] = {
                                "method": "manual-local-source",
                                "source": str(source),
                                "destination": str(destination),
                                "sha256": digest,
                            }
                        else:
                            assert isinstance(source, Path) and isinstance(digest, str)
                            outputs.append(_copy_file(source, destination, digest))
                            acquisition_report[dataset] = {
                                "method": (
                                    "official-download-fixture"
                                    if dataset == "nbaiot"
                                    else "manual-local-source"
                                ),
                                "source": str(source),
                                "destination": str(destination),
                                "sha256": digest,
                            }
                    acquisition_report_path = metadata_root / "acquisition.json"
                    _write_json(acquisition_report_path, {"datasets": acquisition_report})
                    outputs.append(acquisition_report_path)
                    return tuple(outputs)

                def run_extract() -> Sequence[Path]:
                    extraction_root.mkdir(parents=True, exist_ok=True)
                    outputs: list[Path] = []
                    extraction_report: dict[str, object] = {}
                    extraction_report_path = metadata_root / "extraction.json"
                    prior_trees: dict[str, str] = {}
                    if extraction_report_path.is_file():
                        try:
                            prior_payload = json.loads(
                                extraction_report_path.read_text(encoding="utf-8")
                            )
                            prior_datasets = prior_payload["datasets"]
                            if isinstance(prior_datasets, dict):
                                for name, value in prior_datasets.items():
                                    if isinstance(value, dict) and isinstance(
                                        value.get("tree_sha256"), str
                                    ):
                                        prior_trees[str(name)] = value["tree_sha256"]
                        except (OSError, KeyError, TypeError, json.JSONDecodeError):
                            prior_trees = {}
                    for dataset, context in source_context.items():
                        source = acquired[dataset]
                        destination = raw_roots[dataset]
                        kind = str(context["kind"])
                        if destination.is_dir():
                            existing_digest, _size, existing_files = _tree_digest(destination)
                            if prior_trees.get(dataset) != existing_digest:
                                raise RuntimeError(
                                    "Existing extracted source does not match its prior "
                                    f"verified tree: {destination}. Preserve it for "
                                    "inspection and use a new data root."
                                )
                            extraction_report[dataset] = {
                                "extractor": "verified-existing-tree",
                                "destination": str(destination),
                                "files": len(existing_files),
                                "tree_sha256": existing_digest,
                            }
                        elif kind == "directory":
                            digest = str(context["digest"])
                            copied = _copy_tree(source, destination, digest)
                            extraction_report[dataset] = {
                                "extractor": "verified-directory-copy",
                                "destination": str(destination),
                                "files": len(copied),
                                "tree_sha256": _tree_digest(destination)[0],
                            }
                        else:
                            result = _extract_archive(
                                source,
                                destination,
                                kind=kind,
                                dependencies=deps,
                                install_system_tools=options.install_system_tools,
                            )
                            nested_reports: list[dict[str, object]] = []
                            for nested in _nested_rars(destination):
                                nested_destination = nested.with_suffix("")
                                nested_result = deps.rar_extractor(
                                    nested,
                                    nested_destination,
                                    install_system_tools=options.install_system_tools,
                                )
                                nested_reports.append(nested_result.as_dict())
                            extraction_report[dataset] = {
                                **result.as_dict(),
                                "nested_rar": nested_reports,
                                "tree_sha256": _tree_digest(destination)[0],
                            }
                        _reject_excluded_capture_files(destination)
                    _write_json(extraction_report_path, {"datasets": extraction_report})
                    outputs.append(extraction_report_path)
                    return tuple(outputs)

                def run_inspect() -> Sequence[Path]:
                    manifest_root.mkdir(parents=True, exist_ok=True)
                    schema_root.mkdir(parents=True, exist_ok=True)
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
                            acquisition_url=(spec.download_url if dataset == "nbaiot" else None),
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

                stages = (
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
                        lambda: _json_signature({"compute": options.compute}),
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
                                    name: _regular_digest(path)[0]
                                    if path.is_file()
                                    else _tree_digest(path)[0]
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

                for stage in stages:
                    current_stage = stage.name
                    signature = stage.input_signature()
                    if state.reusable(stage.name, signature):
                        reused.append(stage.name)
                    else:
                        state.invalidate_from(stage.name, STAGE_ORDER)
                        outputs = tuple(stage.run())
                        state.complete(stage.name, stage.input_signature(), outputs)
                        executed.append(stage.name)
                    last_completed = stage.name
                    if stage.name == "environment" and compute is None:
                        compute = json.loads(environment_path.read_text(encoding="utf-8"))
                    if stage.name == "inspect":
                        for dataset in options.datasets:
                            source_token = _tree_digest(raw_roots[dataset])[0]
                            manifests[dataset] = str(
                                manifest_root / f"{dataset}-{source_token}.json"
                            )
                            schemas[dataset] = str(
                                schema_root / f"{dataset}-{source_token}.json"
                            )

                return report("sources_verified")
            except BaseException as error:
                failed_stage = current_stage
                return report("failed", error)
    except BaseException as error:
        failed_stage = current_stage
        # Lock acquisition/release failure must not mutate the shared report path;
        # use a process-unique sibling report under the already trusted parent.
        fallback = lock_path.with_name(
            f"{lock_path.stem}-failure-{os.getpid()}-{uuid.uuid4().hex}.json"
        )
        result = {
            "version": REPORT_FORMAT_VERSION,
            "status": "failed",
            "last_completed_stage": last_completed,
            "failed_stage": failed_stage,
            "error": _safe_error(error),
            "recovery_command": _recovery(failed_stage),
            "inputs": {"original": safe_original, "resolved": safe_resolved},
            "report_path": str(fallback),
        }
        _write_json(fallback, result)
        return result
