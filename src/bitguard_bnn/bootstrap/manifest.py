"""Immutable, deterministic provenance manifests for verified dataset sources."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlsplit

from .registry import load_registry
from .types import DatasetSpec


MANIFEST_FORMAT_VERSION = 2
ALLOWED_ACQUISITION_METHODS = frozenset(
    {"official-download", "manual-local-source"}
)
_DIRECTORY_FSYNC_SUPPORTED = os.name != "nt"
_TEMP_CREATE_ATTEMPTS = 16
_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


class SourceManifestError(RuntimeError):
    """Source provenance could not be recorded without losing integrity."""


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceManifestError(f"Manifest field {field} must be a non-empty string.")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field)


def _require_sha256(value: object, field: str) -> str:
    digest = _require_string(value, field)
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SourceManifestError(f"Manifest field {field} is not a lowercase SHA-256.")
    return digest


@dataclass(frozen=True, slots=True)
class SourceFileRecord:
    relative_path: str
    byte_size: int
    mtime_ns: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceFileRecord:
        expected = {"relative_path", "byte_size", "mtime_ns", "sha256"}
        if set(value) != expected:
            raise SourceManifestError("A manifest file record has invalid fields.")
        relative_path = _require_string(value["relative_path"], "relative_path")
        normalized = PurePosixPath(relative_path)
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or "\\" in relative_path
            or normalized.as_posix() != relative_path
            or relative_path in ("", ".")
        ):
            raise SourceManifestError("A manifest file record has an unsafe relative path.")
        byte_size = value["byte_size"]
        if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0:
            raise SourceManifestError("A manifest file record has an invalid byte size.")
        mtime_ns = value["mtime_ns"]
        if not isinstance(mtime_ns, int) or isinstance(mtime_ns, bool) or mtime_ns < 0:
            raise SourceManifestError("A manifest file record has an invalid mtime_ns.")
        return cls(
            relative_path=relative_path,
            byte_size=byte_size,
            mtime_ns=mtime_ns,
            sha256=_require_sha256(value["sha256"], "file.sha256"),
        )


@dataclass(frozen=True, slots=True)
class SourceManifest:
    dataset_name: str
    project_url: str
    doi: str | None
    license_name: str
    acquisition_method: str
    acquisition_url: str | None
    files: tuple[SourceFileRecord, ...]
    total_bytes: int
    content_sha256: str
    version: int = MANIFEST_FORMAT_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "dataset_name": self.dataset_name,
            "project_url": self.project_url,
            "doi": self.doi,
            "license_name": self.license_name,
            "acquisition_method": self.acquisition_method,
            "acquisition_url": self.acquisition_url,
            "files": [record.to_dict() for record in self.files],
            "total_bytes": self.total_bytes,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceManifest:
        expected = {
            "version",
            "dataset_name",
            "project_url",
            "doi",
            "license_name",
            "acquisition_method",
            "acquisition_url",
            "files",
            "total_bytes",
            "content_sha256",
        }
        version = value.get("version")
        if (
            set(value) != expected
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version != MANIFEST_FORMAT_VERSION
        ):
            raise SourceManifestError("Source manifest has invalid fields or version.")
        raw_files = value["files"]
        if not isinstance(raw_files, list) or not all(
            isinstance(item, dict) for item in raw_files
        ):
            raise SourceManifestError("Source manifest files must be a JSON array.")
        files = tuple(SourceFileRecord.from_dict(item) for item in raw_files)
        paths = tuple(record.relative_path for record in files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise SourceManifestError("Source manifest file paths must be unique and ordered.")
        total_bytes = value["total_bytes"]
        if (
            not isinstance(total_bytes, int)
            or isinstance(total_bytes, bool)
            or total_bytes < 0
            or total_bytes != sum(record.byte_size for record in files)
        ):
            raise SourceManifestError("Source manifest total_bytes is inconsistent.")
        method = _require_string(value["acquisition_method"], "acquisition_method")
        if method not in ALLOWED_ACQUISITION_METHODS:
            raise SourceManifestError("Source manifest acquisition method is not allowed.")
        manifest = cls(
            dataset_name=_require_string(value["dataset_name"], "dataset_name"),
            project_url=_require_string(value["project_url"], "project_url"),
            doi=_optional_string(value["doi"], "doi"),
            license_name=_require_string(value["license_name"], "license_name"),
            acquisition_method=method,
            acquisition_url=_optional_string(value["acquisition_url"], "acquisition_url"),
            files=files,
            total_bytes=total_bytes,
            content_sha256=_require_sha256(value["content_sha256"], "content_sha256"),
        )
        official = load_registry().get(manifest.dataset_name)
        if official is None or (
            manifest.project_url,
            manifest.doi,
            manifest.license_name,
        ) != (official.project_url, official.doi, official.license_name):
            raise SourceManifestError(
                "Source manifest metadata does not match the official registry."
            )
        _validate_acquisition_metadata(
            official,
            manifest.acquisition_method,
            manifest.acquisition_url,
        )
        expected_digest = _manifest_content_digest(manifest.to_dict())
        if manifest.content_sha256 != expected_digest:
            raise SourceManifestError("Source manifest content digest is inconsistent.")
        return manifest


def _validate_https_without_credentials(value: str, field: str) -> None:
    try:
        parsed = urlsplit(value)
    except ValueError as error:
        raise SourceManifestError(f"{field} is not a valid HTTPS URL.") from error
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise SourceManifestError(f"{field} must be an HTTPS URL.")
    if parsed.username is not None or parsed.password is not None:
        raise SourceManifestError(f"{field} must not contain credentials.")


def _validate_acquisition_metadata(
    spec: DatasetSpec, method: str, acquisition_url: str | None
) -> None:
    if method not in ALLOWED_ACQUISITION_METHODS:
        raise SourceManifestError(
            f"Acquisition method {method!r} is not one of {sorted(ALLOWED_ACQUISITION_METHODS)}."
        )
    _validate_https_without_credentials(spec.project_url, "project_url")
    official = load_registry().get(spec.name)
    if official is None or official != spec:
        raise SourceManifestError(
            f"Dataset metadata for {spec.name!r} does not match the official registry."
        )
    if spec.name == "botiot":
        if method != "manual-local-source" or acquisition_url is not None:
            raise SourceManifestError(
                "BoT-IoT must use manual-local-source and must never record an acquisition URL."
            )
        return
    if spec.name == "nbaiot":
        if method != "official-download":
            raise SourceManifestError("N-BaIoT must use the official-download method.")
        if acquisition_url is None:
            raise SourceManifestError("N-BaIoT official-download requires its official URL.")
        _validate_https_without_credentials(acquisition_url, "acquisition_url")
        if acquisition_url != spec.download_url:
            raise SourceManifestError(
                "N-BaIoT acquisition URL does not match the official registry."
            )
        return
    raise SourceManifestError(f"Dataset {spec.name!r} is not supported by the registry.")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns if os.name != "nt" else 0,
    )


def _inode_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode))


def _directory_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns if os.name != "nt" else 0,
    )


@dataclass(frozen=True, slots=True)
class _DirectoryPin:
    path: Path
    identity: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _EnumeratedFile:
    relative_path: str
    path: Path
    identity: tuple[int, int, int, int, int, int]
    ancestors: tuple[_DirectoryPin, ...]


@dataclass(frozen=True, slots=True)
class _SourceEnumeration:
    files: tuple[_EnumeratedFile, ...]
    directories: tuple[_DirectoryPin, ...]
    root: _DirectoryPin


def _lstat_directory(path: Path, *, subject: str) -> os.stat_result:
    try:
        current = path.lstat()
    except OSError as error:
        raise SourceManifestError(
            f"Cannot inspect {subject} directory {path}: {error}."
        ) from error
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise SourceManifestError(
            f"{subject.capitalize()} directory {path} became a symlink or non-directory."
        )
    return current


def _validate_ancestor_chain(
    ancestors: tuple[_DirectoryPin, ...], relative_path: str
) -> None:
    for ancestor in ancestors:
        current = _lstat_directory(ancestor.path, subject="source ancestor")
        if _directory_identity(current) != ancestor.identity:
            raise SourceManifestError(
                f"Source ancestor directory {ancestor.path} changed identity while "
                f"processing {relative_path}."
            )


def _validate_enumerated_file(record: _EnumeratedFile, *, phase: str) -> None:
    try:
        current = record.path.lstat()
    except OSError as error:
        raise SourceManifestError(
            f"Source file {record.relative_path} changed {phase}: {error}."
        ) from error
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise SourceManifestError(
            f"Source file {record.relative_path} became a symlink or non-regular file {phase}."
        )
    if _file_identity(current) != record.identity:
        raise SourceManifestError(
            f"Source file {record.relative_path} changed identity {phase}."
        )


def _hash_file(
    path: Path,
) -> tuple[str, tuple[int, int, int, int, int, int]]:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            opened_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_before.st_mode):
                raise SourceManifestError(f"Source file {path} is not regular.")
            opened_identity = _file_identity(opened_before)
            while True:
                chunk = stream.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
            if _file_identity(os.fstat(stream.fileno())) != opened_identity:
                raise SourceManifestError(f"Source file {path} changed while hashing.")
    except SourceManifestError:
        raise
    except OSError as error:
        raise SourceManifestError(f"Cannot hash source file {path}: {error}.") from error
    return digest.hexdigest(), opened_identity


def _enumerate_files(root: Path) -> _SourceEnumeration:
    collected: list[_EnumeratedFile] = []
    directories: list[_DirectoryPin] = []

    def visit(
        directory: Path,
        parts: tuple[str, ...],
        parents: tuple[_DirectoryPin, ...],
        expected_identity: tuple[int, int, int, int, int, int] | None = None,
    ) -> _DirectoryPin:
        directory_stat = _lstat_directory(directory, subject="source ancestor")
        directory_identity = _directory_identity(directory_stat)
        if expected_identity is not None and directory_identity != expected_identity:
            raise SourceManifestError(
                f"Source ancestor directory {directory} changed before enumeration."
            )
        current_pin = _DirectoryPin(directory, directory_identity)
        directories.append(current_pin)
        ancestors = (*parents, current_pin)
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise SourceManifestError(
                f"Cannot enumerate source directory {directory}: {error}."
            ) from error
        for entry in entries:
            relative_parts = (*parts, entry.name)
            relative = PurePosixPath(*relative_parts).as_posix()
            entry_path = directory / entry.name
            try:
                entry_stat = entry_path.lstat()
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise SourceManifestError(
                        f"Source tree contains forbidden symlink {relative}."
                    )
                if stat.S_ISDIR(entry_stat.st_mode):
                    visit(
                        entry_path,
                        relative_parts,
                        ancestors,
                        _directory_identity(entry_stat),
                    )
                elif stat.S_ISREG(entry_stat.st_mode):
                    collected.append(
                        _EnumeratedFile(
                            relative_path=relative,
                            path=entry_path,
                            identity=_file_identity(entry_stat),
                            ancestors=ancestors,
                        )
                    )
                else:
                    raise SourceManifestError(
                        f"Source tree entry {relative} is not a regular file or directory."
                    )
            except OSError as error:
                raise SourceManifestError(
                    f"Cannot inspect source entry {relative}: {error}."
                ) from error

        _validate_ancestor_chain((current_pin,), directory.as_posix())
        return current_pin

    root_pin = visit(root, (), ())
    normalized = tuple(record.relative_path for record in collected)
    if len(normalized) != len(set(normalized)):
        raise SourceManifestError("Source tree contains duplicate normalized relative paths.")
    if normalized != tuple(sorted(normalized)):
        collected.sort(key=lambda record: record.relative_path)
    return _SourceEnumeration(tuple(collected), tuple(directories), root_pin)


def _validate_source_enumeration(enumeration: _SourceEnumeration) -> None:
    for directory in enumeration.directories:
        current = _lstat_directory(directory.path, subject="source snapshot")
        if _directory_identity(current) != directory.identity:
            raise SourceManifestError(
                f"Source directory {directory.path} changed after tree snapshot."
            )
    for source_file in enumeration.files:
        _validate_ancestor_chain(source_file.ancestors, source_file.relative_path)
        _validate_enumerated_file(source_file, phase="after tree snapshot")


def _compare_source_enumerations(
    expected: _SourceEnumeration,
    observed: _SourceEnumeration,
) -> None:
    expected_files = tuple(
        (record.relative_path, record.identity) for record in expected.files
    )
    observed_files = tuple(
        (record.relative_path, record.identity) for record in observed.files
    )
    expected_directories = tuple(
        (record.path, record.identity) for record in expected.directories
    )
    observed_directories = tuple(
        (record.path, record.identity) for record in observed.directories
    )
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise SourceManifestError(
            "Source tree changed between its initial and final ordered snapshots."
        )


def _manifest_content_digest(value: Mapping[str, object]) -> str:
    content = dict(value)
    content.pop("content_sha256", None)
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_manifest(
    source_root: Path | str,
    spec: DatasetSpec,
    *,
    acquisition_method: str,
    acquisition_url: str | None = None,
) -> SourceManifest:
    """Build provenance from an unchanged, symlink-free source directory."""

    _validate_acquisition_metadata(spec, acquisition_method, acquisition_url)
    raw_root = Path(source_root).expanduser()
    try:
        root_lstat = raw_root.lstat()
    except OSError as error:
        raise SourceManifestError(f"Cannot inspect source root {raw_root}: {error}.") from error
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise SourceManifestError("Source root must be a regular non-symlink directory.")
    root = raw_root.resolve(strict=True)
    resolved_root_stat = _lstat_directory(root, subject="source root")
    if _directory_identity(resolved_root_stat) != _directory_identity(root_lstat):
        raise SourceManifestError("Source root changed identity while it was resolved.")
    enumeration = _enumerate_files(root)
    if enumeration.root.identity != _directory_identity(root_lstat):
        raise SourceManifestError("Source root changed identity during enumeration.")
    records: list[SourceFileRecord] = []
    for source_file in enumeration.files:
        _validate_ancestor_chain(source_file.ancestors, source_file.relative_path)
        _validate_enumerated_file(source_file, phase="before hashing")
        digest, opened_identity = _hash_file(source_file.path)
        if opened_identity != source_file.identity:
            raise SourceManifestError(
                f"Source file {source_file.relative_path} opened a different inode while hashing."
            )
        _validate_ancestor_chain(source_file.ancestors, source_file.relative_path)
        _validate_enumerated_file(source_file, phase="after hashing")
        records.append(
            SourceFileRecord(
                source_file.relative_path,
                source_file.identity[3],
                source_file.identity[4],
                digest,
            )
        )
    observed = _enumerate_files(root)
    _compare_source_enumerations(enumeration, observed)
    _validate_source_enumeration(enumeration)
    total_bytes = sum(record.byte_size for record in records)
    provisional = SourceManifest(
        dataset_name=spec.name,
        project_url=spec.project_url,
        doi=spec.doi,
        license_name=spec.license_name,
        acquisition_method=acquisition_method,
        acquisition_url=acquisition_url,
        files=tuple(records),
        total_bytes=total_bytes,
        content_sha256="0" * 64,
    )
    digest = _manifest_content_digest(provisional.to_dict())
    final_observed = _enumerate_files(root)
    _compare_source_enumerations(enumeration, final_observed)
    _validate_source_enumeration(enumeration)
    return SourceManifest(
        dataset_name=provisional.dataset_name,
        project_url=provisional.project_url,
        doi=provisional.doi,
        license_name=provisional.license_name,
        acquisition_method=provisional.acquisition_method,
        acquisition_url=provisional.acquisition_url,
        files=provisional.files,
        total_bytes=provisional.total_bytes,
        content_sha256=digest,
    )


def manifest_json_bytes(manifest: SourceManifest) -> bytes:
    serialized = (
        json.dumps(
            manifest.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        decoded: Any = json.loads(serialized)
        round_trip = SourceManifest.from_dict(decoded)
    except (json.JSONDecodeError, UnicodeError, SourceManifestError) as error:
        raise SourceManifestError(
            f"Source manifest serialization failed validation: {error}."
        ) from error
    if round_trip != manifest:
        raise SourceManifestError("Source manifest serialization did not round-trip exactly.")
    return serialized


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("manifest temporary write made no progress")
        offset += written


def _fsync_parent_directory(path: Path) -> None:
    if not _DIRECTORY_FSYNC_SUPPORTED:
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _existing_manifest_bytes(path: Path) -> bytes | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SourceManifestError(
            f"Cannot safely open source manifest {path}: {error}."
        ) from error
    base_exception_active = False
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceManifestError(
                f"Source manifest {path} must be a regular non-symlink file."
            )
        if before.st_size < 0 or before.st_size > _MAX_MANIFEST_BYTES:
            raise SourceManifestError(
                f"Source manifest {path} exceeds the bounded read limit."
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(_HASH_CHUNK_SIZE, remaining))
            if not chunk:
                raise SourceManifestError(
                    f"Source manifest {path} ended before its recorded size."
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SourceManifestError(
                f"Source manifest {path} grew during its bounded read."
            )
        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError as error:
            raise SourceManifestError(
                f"Source manifest {path} changed during its pinned read: {error}."
            ) from error
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
            raise SourceManifestError(
                f"Source manifest {path} became a symlink or non-regular file."
            )
        if _file_identity(after) != _file_identity(before) or _file_identity(
            current
        ) != _file_identity(before):
            raise SourceManifestError(
                f"Source manifest {path} changed identity during its pinned read."
            )
        return b"".join(chunks)
    except BaseException:
        base_exception_active = True
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as close_error:
            if not base_exception_active:
                raise SourceManifestError(
                    f"Cannot close source manifest {path}: {close_error}."
                ) from close_error


def _lstat_manifest_file(path: Path, *, subject: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise SourceManifestError(f"Cannot inspect {subject} {path}: {error}.") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise SourceManifestError(f"{subject.capitalize()} {path} is not a regular file.")
    return value


def _unlink_owned_manifest_path(
    path: Path,
    expected_identity: tuple[int, int, int],
) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceManifestError(
            f"Cannot inspect private manifest path {path}: {error}."
        ) from error
    if _inode_identity(current) != expected_identity:
        raise SourceManifestError(
            f"Private manifest path {path} changed identity and was preserved."
        )
    try:
        path.unlink()
    except OSError as error:
        raise SourceManifestError(
            f"Cannot remove private manifest path {path}: {error}."
        ) from error


def _create_temporary(path: Path) -> tuple[Path, int]:
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    for _attempt in range(_TEMP_CREATE_ATTEMPTS):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            return temporary, os.open(temporary, flags, 0o600)
        except FileExistsError:
            continue
        except OSError as error:
            raise SourceManifestError(
                f"Cannot create private source manifest temporary file: {error}."
            ) from error
    raise SourceManifestError("Cannot reserve a unique source manifest temporary file.")


def _pin_manifest_temporary(
    temporary: Path,
    descriptor: int,
    expected_identity: tuple[int, int, int],
) -> Path:
    for _attempt in range(_TEMP_CREATE_ATTEMPTS):
        pin = temporary.with_name(f"{temporary.name}.{uuid.uuid4().hex}.pin")
        try:
            os.link(temporary, pin)
        except FileExistsError:
            continue
        except OSError as error:
            raise SourceManifestError(
                f"Cannot create private manifest pin for {temporary}: {error}."
            ) from error
        pin_stat = _lstat_manifest_file(pin, subject="private manifest pin")
        pin_identity = _inode_identity(pin_stat)
        try:
            temporary_stat = _lstat_manifest_file(
                temporary,
                subject="manifest temporary",
            )
            temporary_identity = _inode_identity(temporary_stat)
            descriptor_identity = _inode_identity(os.fstat(descriptor))
        except (OSError, SourceManifestError):
            _unlink_owned_manifest_path(pin, pin_identity)
            raise
        if (
            pin_identity != expected_identity
            or temporary_identity != expected_identity
            or descriptor_identity != expected_identity
        ):
            _unlink_owned_manifest_path(pin, pin_identity)
            raise SourceManifestError(
                f"Manifest temporary {temporary} changed identity while creating its pin."
            )
        return pin
    raise SourceManifestError("Cannot reserve a unique private manifest pin.")


def write_source_manifest(path: Path | str, manifest: SourceManifest) -> bool:
    """Publish stable JSON without ever replacing an existing manifest.

    Returns ``True`` for a new publication and ``False`` for byte-identical reuse.
    """

    payload = manifest_json_bytes(manifest)
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise SourceManifestError("Serialized source manifest exceeds the write limit.")
    raw = Path(path).expanduser()
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as error:
        raise SourceManifestError(
            f"Source manifest parent {raw.parent} must already exist: {error}."
        ) from error
    target = parent / raw.name
    existing = _existing_manifest_bytes(target)
    if existing is not None:
        if existing == payload:
            return False
        raise SourceManifestError(
            f"Source manifest {target} already exists with different bytes; refusing to overwrite."
        )

    temporary, descriptor = _create_temporary(target)
    temporary_identity: tuple[int, int, int] | None = None
    pin: Path | None = None
    prepared = False
    base_exception_active = False
    try:
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise SourceManifestError(
                    f"Manifest temporary {temporary} is not a regular file."
                )
            temporary_identity = _inode_identity(opened)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            written = os.fstat(descriptor)
            if (
                _inode_identity(written) != temporary_identity
                or written.st_size != len(payload)
            ):
                raise SourceManifestError(
                    f"Manifest temporary {temporary} changed while it was written."
                )
            pin = _pin_manifest_temporary(
                temporary,
                descriptor,
                temporary_identity,
            )
        except OSError as error:
            raise SourceManifestError(
                f"Cannot durably prepare source manifest temporary: {error}."
            ) from error
        except BaseException:
            base_exception_active = True
            raise
        finally:
            try:
                os.close(descriptor)
            except OSError as close_error:
                if not base_exception_active:
                    raise SourceManifestError(
                        f"Cannot close source manifest temporary: {close_error}."
                    ) from close_error

        assert temporary_identity is not None
        assert pin is not None
        _unlink_owned_manifest_path(temporary, temporary_identity)
        prepared = True
    finally:
        if not prepared:
            for candidate in (pin, temporary):
                if candidate is None or temporary_identity is None:
                    continue
                try:
                    _unlink_owned_manifest_path(candidate, temporary_identity)
                except Exception:
                    pass

    assert pin is not None
    assert temporary_identity is not None

    try:
        os.link(pin, target)
    except FileExistsError as error:
        try:
            existing = _existing_manifest_bytes(target)
        finally:
            try:
                _unlink_owned_manifest_path(pin, temporary_identity)
            except SourceManifestError:
                pass
        if existing == payload:
            return False
        raise SourceManifestError(
            f"Source manifest {target} appeared concurrently with different bytes; "
            "refusing to overwrite."
        ) from error
    except OSError as error:
        try:
            _unlink_owned_manifest_path(pin, temporary_identity)
        except SourceManifestError:
            pass
        raise SourceManifestError(
            f"Cannot publish source manifest {target} without overwrite: {error}."
        ) from error

    try:
        final_stat = _lstat_manifest_file(target, subject="published source manifest")
    except SourceManifestError:
        try:
            _unlink_owned_manifest_path(pin, temporary_identity)
        except SourceManifestError:
            pass
        raise
    final_identity = _inode_identity(final_stat)
    if final_identity != temporary_identity:
        try:
            _unlink_owned_manifest_path(target, final_identity)
        except SourceManifestError:
            pass
        try:
            _unlink_owned_manifest_path(pin, temporary_identity)
        except SourceManifestError:
            pass
        raise SourceManifestError(
            f"Published source manifest {target} does not have the verified pin identity."
        )

    try:
        _fsync_parent_directory(target)
        _unlink_owned_manifest_path(pin, temporary_identity)
        _fsync_parent_directory(target)
    except (OSError, SourceManifestError) as error:
        raise SourceManifestError(
            f"Source manifest {target} was published, but durable cleanup failed: {error}."
        ) from error
    return True
