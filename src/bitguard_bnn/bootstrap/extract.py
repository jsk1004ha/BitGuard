"""Archive extraction with path, identity, size, and publication boundaries."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
import uuid
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from bitguard_bnn.bootstrap.fsops import rename_directory_noreplace


class ArchiveExtractionError(RuntimeError):
    """An archive cannot be extracted without violating the safety contract."""


class MissingArchiveToolError(ArchiveExtractionError):
    """A supported external extractor is missing and system changes were not allowed."""

    def __init__(self, command: Sequence[str]) -> None:
        self.command = tuple(command)
        super().__init__(
            "7-Zip-compatible extraction is required. Run exactly: "
            + " ".join(self.command)
        )


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    path: str
    size: int
    is_dir: bool

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "is_dir": self.is_dir}


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    source: str
    destination: str
    extractor: str
    files: tuple[str, ...]
    total_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "destination": self.destination,
            "extractor": self.extractor,
            "files": list(self.files),
            "total_bytes": self.total_bytes,
        }


def _reject_capture_entries(entries: Sequence[ArchiveEntry]) -> None:
    excluded = tuple(
        entry.path
        for entry in entries
        if not entry.is_dir
        and PurePosixPath(entry.path).suffix.casefold() in {".pcap", ".pcapng"}
    )
    if excluded:
        sample = ", ".join(excluded[:3])
        raise ArchiveExtractionError(
            "PCAP capture input is excluded from the CSV bootstrap; "
            f"archive listing contains: {sample}"
        )


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_SUPPORTED_ZIP_METHODS = {
    zipfile.ZIP_STORED,
    zipfile.ZIP_DEFLATED,
    zipfile.ZIP_BZIP2,
    zipfile.ZIP_LZMA,
}
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_DEFAULT_MAX_EXPANDED_BYTES = 1 << 40
_DEFAULT_CHUNK_SIZE = 1 << 20


def _is_reparse(result: os.stat_result) -> bool:
    attributes = getattr(result, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _safe_relative_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ArchiveExtractionError(f"unsafe archive path: {raw!r}")
    if "\\" in raw or raw.startswith("/") or _WINDOWS_DRIVE.match(raw):
        raise ArchiveExtractionError(f"unsafe archive path: {raw!r}")
    raw_parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ArchiveExtractionError(f"unsafe archive path: {raw!r}")

    normalized_parts: list[str] = []
    for raw_part in raw_parts:
        part = unicodedata.normalize("NFC", raw_part)
        base = part.rstrip(" .").split(".", 1)[0].rstrip(" ").casefold()
        if (
            not part
            or part.endswith((" ", "."))
            or ":" in part
            or any(character in '<>"|?*' or ord(character) < 32 for character in part)
            or base in _WINDOWS_RESERVED
        ):
            raise ArchiveExtractionError(f"unsafe archive path: {raw!r}")
        normalized_parts.append(part)
    normalized = PurePosixPath(*normalized_parts).as_posix()
    if PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:
        raise ArchiveExtractionError(f"unsafe archive path: {raw!r}")
    return normalized


def _validate_entries(entries: Sequence[ArchiveEntry]) -> tuple[ArchiveEntry, ...]:
    seen: dict[str, str] = {}
    validated: list[ArchiveEntry] = []
    for entry in entries:
        if isinstance(entry.size, bool) or not isinstance(entry.size, int) or entry.size < 0:
            raise ArchiveExtractionError(f"invalid declared size for {entry.path!r}")
        normalized = _safe_relative_path(entry.path)
        key = normalized.casefold()
        if key in seen:
            raise ArchiveExtractionError(
                "duplicate archive destination: "
                f"{entry.path!r} conflicts with {seen[key]!r}"
            )
        seen[key] = entry.path
        validated.append(ArchiveEntry(normalized, entry.size, entry.is_dir))
    by_key = {entry.path.casefold(): entry for entry in validated}
    for entry in validated:
        for parent in PurePosixPath(entry.path).parents:
            ancestor = by_key.get(parent.as_posix().casefold())
            if ancestor is not None and not ancestor.is_dir:
                raise ArchiveExtractionError(
                    "archive file conflicts with a descendant destination: "
                    f"{ancestor.path!r} and {entry.path!r}"
                )
    return tuple(validated)


def _lstat(path: Path, description: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise ArchiveExtractionError(f"cannot inspect {description} {path}: {error}") from error


def _ensure_secure_existing_directory(path: Path) -> os.stat_result:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        result = _lstat(current, "destination parent")
        if not stat.S_ISDIR(result.st_mode) or stat.S_ISLNK(result.st_mode) or _is_reparse(result):
            raise ArchiveExtractionError(
                f"destination parent contains a link, reparse point, or non-directory: {current}"
            )
    return _lstat(absolute, "destination parent")


def _path_lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ArchiveExtractionError(f"cannot inspect destination {path}: {error}") from error
    return True


def _prepare_destination(destination: Path) -> tuple[Path, os.stat_result]:
    absolute = destination.expanduser().absolute()
    if _path_lexists(absolute):
        raise ArchiveExtractionError(f"destination must not already exist: {absolute}")
    parent_result = _ensure_secure_existing_directory(absolute.parent)
    return absolute, parent_result


def _open_regular_source(path: Path) -> tuple[BinaryIO, os.stat_result]:
    before = _lstat(path, "archive source")
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or _is_reparse(before):
        raise ArchiveExtractionError(f"archive source must be a regular non-link file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArchiveExtractionError(f"cannot open archive source {path}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_object(before, opened):
            raise ArchiveExtractionError(f"archive source changed while opening: {path}")
        return os.fdopen(descriptor, "rb"), opened
    except BaseException:
        os.close(descriptor)
        raise


def _verify_source_unchanged(path: Path, opened: os.stat_result, handle: BinaryIO) -> None:
    current_fd = os.fstat(handle.fileno())
    try:
        current_path = path.lstat()
    except OSError as error:
        raise ArchiveExtractionError(f"archive source changed during extraction: {path}") from error
    if (
        not _same_object(opened, current_fd)
        or not _same_object(opened, current_path)
        or current_fd.st_size != opened.st_size
        or current_fd.st_mtime_ns != opened.st_mtime_ns
        or current_path.st_size != opened.st_size
        or current_path.st_mtime_ns != opened.st_mtime_ns
    ):
        raise ArchiveExtractionError(f"archive source changed during extraction: {path}")


def _declared_preflight(
    entries: Sequence[ArchiveEntry],
    parent: Path,
    *,
    max_total_bytes: int,
    disk_free_fn: Callable[[Path], int] | None,
) -> int:
    if (
        isinstance(max_total_bytes, bool)
        or not isinstance(max_total_bytes, int)
        or max_total_bytes < 0
    ):
        raise ValueError("max_total_bytes must be a non-negative integer")
    declared = sum(entry.size for entry in entries if not entry.is_dir)
    if declared > max_total_bytes:
        raise ArchiveExtractionError(
            f"declared expanded size exceeds configured limit: declared={declared}, "
            f"limit={max_total_bytes}"
        )
    try:
        available = (
            shutil.disk_usage(parent).free if disk_free_fn is None else disk_free_fn(parent)
        )
    except Exception as error:
        raise ArchiveExtractionError(f"cannot inspect extraction disk space: {error}") from error
    if isinstance(available, bool) or not isinstance(available, int) or available < 0:
        raise ArchiveExtractionError(f"invalid available disk byte count: {available}")
    if declared > available:
        raise ArchiveExtractionError(
            f"insufficient disk for declared extraction: required={declared}, available={available}"
        )
    return declared


def _private_staging(
    parent: Path, parent_expected: os.stat_result
) -> tuple[Path, os.stat_result]:
    parent_before = _lstat(parent, "destination parent")
    if not _same_object(parent_expected, parent_before):
        raise ArchiveExtractionError("destination parent changed before staging creation")
    try:
        staging = Path(tempfile.mkdtemp(prefix=".bitguard-extract-", dir=parent))
        os.chmod(staging, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        result = staging.lstat()
    except OSError as error:
        raise ArchiveExtractionError(f"cannot create private extraction root: {error}") from error
    if not stat.S_ISDIR(result.st_mode) or stat.S_ISLNK(result.st_mode) or _is_reparse(result):
        raise ArchiveExtractionError("private extraction root is not an isolated directory")
    parent_after = _lstat(parent, "destination parent")
    if not _same_object(parent_expected, parent_after):
        _cleanup_staging(staging, result)
        raise ArchiveExtractionError("destination parent changed during staging creation")
    return staging, result


def _assert_root_identity(root: Path, expected: os.stat_result) -> None:
    current = _lstat(root, "private extraction root")
    if not _same_object(expected, current) or not stat.S_ISDIR(current.st_mode):
        raise ArchiveExtractionError("private extraction root was replaced during extraction")


def _contained(root: Path, relative: str) -> Path:
    destination = root.joinpath(*PurePosixPath(relative).parts)
    resolved = destination.resolve(strict=False)
    root_resolved = root.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ArchiveExtractionError(
            f"archive destination escapes private root: {relative!r}"
        ) from error
    return destination


def _ensure_private_directory(
    root: Path,
    root_expected: os.stat_result,
    relative: str,
) -> tuple[Path, os.stat_result]:
    current = root
    if relative not in {"", "."}:
        for component in PurePosixPath(relative).parts:
            _assert_root_identity(root, root_expected)
            current /= component
            try:
                os.mkdir(current, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            except FileExistsError:
                pass
            result = _lstat(current, "private extraction directory")
            if (
                not stat.S_ISDIR(result.st_mode)
                or stat.S_ISLNK(result.st_mode)
                or _is_reparse(result)
            ):
                raise ArchiveExtractionError(
                    f"unsafe private extraction directory: {relative!r}"
                )
    result = _lstat(current, "private extraction directory")
    return current, result


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _regular_sha256(
    path: Path, expected: os.stat_result | None = None
) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArchiveExtractionError(
            f"cannot verify staged extraction file {path}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (expected is not None and not _same_object(expected, before))
        ):
            raise ArchiveExtractionError(f"staged extraction file identity changed: {path}")
        digest = hashlib.sha256()
        while block := os.read(descriptor, _DEFAULT_CHUNK_SIZE):
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            not _same_object(before, after)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ArchiveExtractionError(f"staged extraction file changed while hashing: {path}")
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def _write_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
    *,
    chunk_size: int,
    remaining_total: int,
    parent_expected: os.stat_result,
) -> int:
    parent_before = _lstat(destination.parent, "private extraction directory")
    if not _same_object(parent_expected, parent_before):
        raise ArchiveExtractionError("private extraction directory changed before file creation")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    destination_identity: os.stat_result | None = None
    digest = hashlib.sha256()
    written = 0
    try:
        descriptor = os.open(destination, flags, stat.S_IRUSR | stat.S_IWUSR)
        parent_after_open = _lstat(destination.parent, "private extraction directory")
        if not _same_object(parent_expected, parent_after_open):
            raise ArchiveExtractionError(
                "private extraction directory changed during file creation"
            )
        destination_identity = os.fstat(descriptor)
        with archive.open(info, "r") as source:
            while True:
                block = source.read(chunk_size)
                if not block:
                    break
                written += len(block)
                if written > info.file_size or written > remaining_total:
                    raise ArchiveExtractionError(
                        f"archive member exceeded declared byte limit: {info.filename!r}"
                    )
                view = memoryview(block)
                while view:
                    count = os.write(descriptor, view)
                    if count <= 0:
                        raise ArchiveExtractionError(
                            f"short write while extracting {info.filename!r}"
                        )
                    view = view[count:]
                digest.update(block)
        if written != info.file_size:
            raise ArchiveExtractionError(
                f"archive member size mismatch for {info.filename!r}: "
                f"declared={info.file_size}, actual={written}"
            )
        os.fsync(descriptor)
        descriptor_result = os.fstat(descriptor)
        if (
            destination_identity is None
            or not _same_object(destination_identity, descriptor_result)
            or descriptor_result.st_size != written
        ):
            raise ArchiveExtractionError("staged extraction file changed while writing")
        os.close(descriptor)
        descriptor = -1
        destination_result = destination.lstat()
        if destination_identity is None or not _same_object(
            destination_identity, destination_result
        ):
            raise ArchiveExtractionError("staged extraction file identity mismatch")
        staged_digest, staged_result = _regular_sha256(
            destination, destination_identity
        )
        if staged_result.st_size != written or staged_digest != digest.hexdigest():
            raise ArchiveExtractionError("staged extraction file content changed")
        _fsync_directory(destination.parent)
        return written
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _zip_entries(infos: Sequence[zipfile.ZipInfo]) -> tuple[ArchiveEntry, ...]:
    raw: list[ArchiveEntry] = []
    for info in infos:
        is_dir = info.is_dir()
        raw_name = info.filename[:-1] if is_dir and info.filename.endswith("/") else info.filename
        if info.flag_bits & 0x1:
            raise ArchiveExtractionError(f"encrypted archive entry is unsupported: {raw_name!r}")
        if info.compress_type not in _SUPPORTED_ZIP_METHODS:
            raise ArchiveExtractionError(
                f"unsupported ZIP compression method for {raw_name!r}: {info.compress_type}"
            )
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if file_type and (
            (is_dir and file_type != stat.S_IFDIR)
            or (not is_dir and file_type != stat.S_IFREG)
        ):
            raise ArchiveExtractionError(
                f"archive link or special entry is unsupported: {raw_name!r}"
            )
        if is_dir and info.file_size:
            raise ArchiveExtractionError(
                f"archive directory declares a non-zero size: {raw_name!r}"
            )
        raw.append(ArchiveEntry(raw_name, info.file_size, is_dir))
    return _validate_entries(raw)


def _cleanup_staging(staging: Path, expected: os.stat_result) -> None:
    try:
        current = staging.lstat()
    except FileNotFoundError:
        return
    if not _same_object(expected, current) or not stat.S_ISDIR(current.st_mode):
        raise ArchiveExtractionError(
            f"refusing to clean a replaced private extraction root: {staging}"
        )
    shutil.rmtree(staging)


def _attach_cleanup_context(primary: BaseException, message: str) -> None:
    """Attach recovery detail without relying on Python 3.11 exception notes."""

    try:
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            add_note(message)
            return
        existing = getattr(primary, "__bitguard_cleanup_notes__", ())
        if not isinstance(existing, tuple):
            existing = ()
        setattr(primary, "__bitguard_cleanup_notes__", (*existing, message))
    except Exception:
        # Cleanup diagnostics must never replace the exception already unwinding.
        return


def _cleanup_staging_after_operation(
    staging: Path, expected: os.stat_result
) -> None:
    """Clean staging without replacing an exception already leaving extraction."""

    primary = sys.exc_info()[1]
    try:
        _cleanup_staging(staging, expected)
    except Exception as cleanup_error:
        recovery = (
            "private extraction cleanup failed; retained staging may remain at "
            f"{staging}. Inspect it and remove it manually after confirming the path "
            "is safe. Cleanup error: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
        if primary is not None:
            _attach_cleanup_context(primary, recovery)
            return
        raise ArchiveExtractionError(recovery) from cleanup_error


def _publish_directory(
    source: Path,
    source_identity: os.stat_result,
    destination: Path,
    destination_parent_identity: os.stat_result,
) -> None:
    _assert_root_identity(source, source_identity)
    parent_now = _lstat(destination.parent, "destination parent")
    if not _same_object(destination_parent_identity, parent_now):
        raise ArchiveExtractionError("destination parent changed before publication")
    try:
        rename_directory_noreplace(source, destination)
    except FileExistsError as error:
        raise ArchiveExtractionError(
            f"destination appeared before no-clobber publication: {destination}"
        ) from error
    except OSError as error:
        raise ArchiveExtractionError(
            f"atomic extraction publication failed for {destination}: {error}; "
            "all observed paths were preserved"
        ) from error
    published = _lstat(destination, "published extraction directory")
    if not _same_object(source_identity, published) or not stat.S_ISDIR(
        published.st_mode
    ):
        raise ArchiveExtractionError(
            f"published extraction directory identity changed: {destination}; "
            "the observed path was preserved"
        )
    _fsync_directory(destination)
    _fsync_directory(destination.parent)


def extract_zip(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    max_total_bytes: int = _DEFAULT_MAX_EXPANDED_BYTES,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    disk_free_fn: Callable[[Path], int] | None = None,
) -> ExtractionResult:
    """Validate an entire ZIP, stream it privately, then publish without clobbering."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    source_path = Path(source).expanduser().absolute()
    destination_path, parent_identity = _prepare_destination(Path(destination))
    handle, source_identity = _open_regular_source(source_path)
    staging: Path | None = None
    staging_identity: os.stat_result | None = None
    try:
        try:
            archive = zipfile.ZipFile(handle)
        except (OSError, EOFError, UnicodeError, zipfile.BadZipFile) as error:
            raise ArchiveExtractionError(f"invalid ZIP archive {source_path}: {error}") from error
        with archive:
            infos = archive.infolist()
            entries = _zip_entries(infos)
            _reject_capture_entries(entries)
            declared = _declared_preflight(
                entries,
                destination_path.parent,
                max_total_bytes=max_total_bytes,
                disk_free_fn=disk_free_fn,
            )
            staging, staging_identity = _private_staging(
                destination_path.parent, parent_identity
            )
            by_path = {entry.path: info for entry, info in zip(entries, infos)}
            actual = 0
            for entry in entries:
                _assert_root_identity(staging, staging_identity)
                if entry.is_dir:
                    _ensure_private_directory(
                        staging, staging_identity, entry.path
                    )
                    continue
                parent_relative = PurePosixPath(entry.path).parent.as_posix()
                parent, parent_expected = _ensure_private_directory(
                    staging, staging_identity, parent_relative
                )
                target = parent / PurePosixPath(entry.path).name
                actual += _write_zip_member(
                    archive,
                    by_path[entry.path],
                    target,
                    chunk_size=chunk_size,
                    remaining_total=declared - actual,
                    parent_expected=parent_expected,
                )
            if actual != declared:
                raise ArchiveExtractionError(
                    f"archive expanded size mismatch: declared={declared}, actual={actual}"
                )
        _verify_source_unchanged(source_path, source_identity, handle)
        assert staging is not None and staging_identity is not None
        _publish_directory(staging, staging_identity, destination_path, parent_identity)
        return ExtractionResult(
            source=str(source_path),
            destination=str(destination_path),
            extractor="zipfile",
            files=tuple(sorted(entry.path for entry in entries if not entry.is_dir)),
            total_bytes=declared,
        )
    except (
        ArchiveExtractionError,
        EOFError,
        UnicodeError,
        zipfile.BadZipFile,
        RuntimeError,
        OSError,
    ) as error:
        if isinstance(error, ArchiveExtractionError):
            raise
        raise ArchiveExtractionError(f"ZIP extraction failed for {source_path}: {error}") from error
    finally:
        handle.close()
        if staging is not None and staging_identity is not None:
            _cleanup_staging_after_operation(staging, staging_identity)


def parse_7z_listing(output: str) -> tuple[ArchiveEntry, ...]:
    """Parse technical 7-Zip listing output and reject ambiguous entry metadata."""

    if not isinstance(output, str):
        raise TypeError("7-Zip listing output must be text")
    marker = re.search(r"(?m)^-{5,}\s*$", output)
    if marker is None:
        raise ArchiveExtractionError("7-Zip listing is missing its entry delimiter")
    body = output[marker.end() :]
    blocks = re.split(r"\r?\n\s*\r?\n", body.strip()) if body.strip() else []
    raw: list[ArchiveEntry] = []
    for block in blocks:
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
            elif line.endswith(" ="):
                key, value = line[:-2], ""
            else:
                raise ArchiveExtractionError(f"ambiguous 7-Zip listing line: {line!r}")
            if key in fields:
                raise ArchiveExtractionError(f"duplicate 7-Zip listing field: {key!r}")
            fields[key] = value
        path = fields.get("Path")
        size_text = fields.get("Size")
        folder = fields.get("Folder")
        if path is None or size_text is None or folder not in {"+", "-"}:
            raise ArchiveExtractionError("ambiguous 7-Zip listing entry metadata")
        try:
            size = int(size_text)
        except ValueError as error:
            raise ArchiveExtractionError(f"invalid Size in 7-Zip listing: {size_text!r}") from error
        if fields.get("Encrypted", "-") != "-":
            raise ArchiveExtractionError(f"encrypted archive entry is unsupported: {path!r}")
        mode = fields.get("Mode", "")
        attributes = fields.get("Attributes", "")
        attribute_mode = next(
            (token for token in reversed(attributes.split()) if token[:1] in "-dlbcps"),
            "",
        )
        effective_mode = mode or attribute_mode
        if (
            fields.get("Symbolic Link", "")
            or fields.get("Hard Link", "")
            or fields.get("Anti", "-") == "+"
            or (effective_mode and effective_mode[0] not in {"-", "d"})
            or (folder == "+" and effective_mode.startswith("-"))
            or (folder == "-" and effective_mode.startswith("d"))
        ):
            raise ArchiveExtractionError(f"archive link or special entry is unsupported: {path!r}")
        if folder == "+" and size:
            raise ArchiveExtractionError(
                f"archive directory declares a non-zero size: {path!r}"
            )
        raw.append(ArchiveEntry(path, size, folder == "+"))
    if not raw:
        raise ArchiveExtractionError("7-Zip listing contains no archive entries")
    return _validate_entries(raw)


def _remediation_command(
    platform_name: str,
    which_fn: Callable[[str], str | None],
    package_manager: str | None,
) -> tuple[str, ...]:
    system = platform_name.casefold()
    if system.startswith("win"):
        return (
            "winget",
            "install",
            "--id",
            "7zip.7zip",
            "--exact",
            "--source",
            "winget",
            "--silent",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--disable-interactivity",
        )
    manager = package_manager
    if manager is None:
        manager = next((name for name in ("apt-get", "dnf") if which_fn(name)), None)
    if manager == "apt-get":
        return ("apt-get", "install", "-y", "p7zip-full")
    if manager == "dnf":
        return ("dnf", "install", "-y", "p7zip", "p7zip-plugins")
    raise MissingArchiveToolError(("apt-get", "install", "-y", "p7zip-full"))


def _find_or_install_7z(
    *,
    install_system_tools: bool,
    which_fn: Callable[[str], str | None],
    run_fn: Callable[..., Any],
    platform_name: str,
    package_manager: str | None,
) -> str:
    found = _find_7z_executable(which_fn, platform_name)
    if found:
        return found
    command = _remediation_command(platform_name, which_fn, package_manager)
    if not install_system_tools:
        raise MissingArchiveToolError(command)
    executable = which_fn(command[0])
    invocation = [executable or command[0], *command[1:]]
    try:
        result = run_fn(
            invocation,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=900,
        )
    except Exception as error:
        raise ArchiveExtractionError(
            f"7-Zip installation command failed: {type(error).__name__}"
        ) from error
    if getattr(result, "returncode", 1) != 0:
        raise ArchiveExtractionError(
            "7-Zip installation command failed with exit code "
            f"{getattr(result, 'returncode', None)}"
        )
    found = _find_7z_executable(which_fn, platform_name)
    if not found:
        raise ArchiveExtractionError(
            "7-Zip installation completed but no compatible executable was found"
        )
    return found


def _find_7z_executable(
    which_fn: Callable[[str], str | None], platform_name: str
) -> str | None:
    for name in ("7z", "7zz", "7za"):
        found = which_fn(name)
        if found:
            return found
    if platform_name.casefold().startswith("win"):
        for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            root = os.environ.get(variable)
            if not root:
                continue
            candidate = Path(root) / "7-Zip" / "7z.exe"
            try:
                if candidate.is_file():
                    return str(candidate)
            except OSError:
                continue
    return None


def _run_7z(
    run_fn: Callable[..., Any], args: list[str], *, operation: str, timeout: int
) -> Any:
    try:
        result = run_fn(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            timeout=timeout,
        )
    except Exception as error:
        raise ArchiveExtractionError(
            f"7-Zip {operation} failed before completion: {type(error).__name__}"
        ) from error
    if getattr(result, "returncode", 1) != 0:
        raise ArchiveExtractionError(
            f"7-Zip {operation} failed with exit code {getattr(result, 'returncode', None)}"
        )
    return result


def _validate_result_tree(staging: Path, entries: Sequence[ArchiveEntry]) -> int:
    implicit_directories = {
        parent.as_posix()
        for entry in entries
        for parent in PurePosixPath(entry.path).parents
        if parent.as_posix() != "."
    }
    observed: dict[str, tuple[bool, int]] = {}
    observed_keys: set[str] = set()
    for base, directories, files in os.walk(staging, topdown=True, followlinks=False):
        base_path = Path(base)
        for name in [*directories, *files]:
            path = base_path / name
            result = path.lstat()
            relative = path.relative_to(staging).as_posix()
            normalized = _safe_relative_path(relative)
            is_dir = stat.S_ISDIR(result.st_mode)
            if (
                stat.S_ISLNK(result.st_mode)
                or _is_reparse(result)
                or not (is_dir or stat.S_ISREG(result.st_mode))
            ):
                raise ArchiveExtractionError(
                    f"unsafe link or special file in 7-Zip result tree: {relative!r}"
                )
            key = normalized.casefold()
            if key in observed_keys:
                raise ArchiveExtractionError(f"duplicate path in 7-Zip result tree: {relative!r}")
            observed_keys.add(key)
            observed[normalized] = (is_dir, 0 if is_dir else result.st_size)
    expected_view = {
        **{path: (True, 0) for path in implicit_directories},
        **{entry.path: (entry.is_dir, entry.size) for entry in entries},
    }
    if observed != expected_view:
        raise ArchiveExtractionError(
            "7-Zip result tree does not match its validated listing: "
            f"expected={sorted(expected_view)}, observed={sorted(observed)}"
        )
    return sum(size for is_dir, size in observed.values() if not is_dir)


def extract_rar(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    install_system_tools: bool = False,
    max_total_bytes: int = _DEFAULT_MAX_EXPANDED_BYTES,
    disk_free_fn: Callable[[Path], int] | None = None,
    which_fn: Callable[[str], str | None] = shutil.which,
    run_fn: Callable[..., Any] = subprocess.run,
    platform_name: str | None = None,
    package_manager: str | None = None,
) -> ExtractionResult:
    """List and validate a RAR with 7-Zip, extract privately, then revalidate."""

    source_path = Path(source).expanduser().absolute()
    destination_path, parent_identity = _prepare_destination(Path(destination))
    handle, source_identity = _open_regular_source(source_path)
    handle.close()
    tool = _find_or_install_7z(
        install_system_tools=install_system_tools,
        which_fn=which_fn,
        run_fn=run_fn,
        platform_name=platform_name or platform.system(),
        package_manager=package_manager,
    )
    staging, staging_identity = _private_staging(destination_path.parent, parent_identity)
    pinned_source = staging / f"source-{uuid.uuid4().hex}.rar"
    extraction_root = staging / "content"
    try:
        try:
            os.link(source_path, pinned_source, follow_symlinks=False)
        except OSError as error:
            raise ArchiveExtractionError(
                f"cannot identity-pin RAR source for external extraction: {error}"
            ) from error
        pinned_result = pinned_source.lstat()
        if not _same_object(source_identity, pinned_result):
            raise ArchiveExtractionError("RAR source identity changed while it was pinned")
        listing_result = _run_7z(
            run_fn,
            [tool, "l", "-slt", "-sccUTF-8", "--", str(pinned_source)],
            operation="listing",
            timeout=300,
        )
        stdout = getattr(listing_result, "stdout", "")
        if not isinstance(stdout, str) or len(stdout) > 64 * 1024 * 1024:
            raise ArchiveExtractionError("7-Zip listing output is invalid or exceeds 64 MiB")
        try:
            entries = parse_7z_listing(stdout)
        except ArchiveExtractionError as error:
            raise ArchiveExtractionError(f"7-Zip listing failed validation: {error}") from error
        _reject_capture_entries(entries)
        declared = _declared_preflight(
            entries,
            destination_path.parent,
            max_total_bytes=max_total_bytes,
            disk_free_fn=disk_free_fn,
        )
        extraction_root.mkdir(mode=stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        extraction_identity = extraction_root.lstat()
        _run_7z(
            run_fn,
            [
                tool,
                "x",
                "-y",
                "-bd",
                "-bb0",
                "-sccUTF-8",
                f"-o{extraction_root}",
                "--",
                str(pinned_source),
            ],
            operation="extraction",
            timeout=24 * 60 * 60,
        )
        _assert_root_identity(extraction_root, extraction_identity)
        actual = _validate_result_tree(extraction_root, entries)
        if actual > declared or actual > max_total_bytes:
            raise ArchiveExtractionError(
                "7-Zip result exceeded validated byte ceiling: "
                f"declared={declared}, actual={actual}"
            )
        source_now = _lstat(source_path, "RAR source")
        if (
            not _same_object(source_identity, source_now)
            or source_now.st_size != source_identity.st_size
            or source_now.st_mtime_ns != source_identity.st_mtime_ns
        ):
            raise ArchiveExtractionError("RAR source changed during extraction")
        _publish_directory(
            extraction_root,
            extraction_identity,
            destination_path,
            parent_identity,
        )
        return ExtractionResult(
            source=str(source_path),
            destination=str(destination_path),
            extractor=Path(tool).name,
            files=tuple(sorted(entry.path for entry in entries if not entry.is_dir)),
            total_bytes=actual,
        )
    finally:
        _cleanup_staging_after_operation(staging, staging_identity)
