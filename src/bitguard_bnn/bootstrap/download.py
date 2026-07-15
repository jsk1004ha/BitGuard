"""Crash-safe, resumable HTTP downloads for bootstrap source archives."""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import SplitResult, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 300.0
USER_AGENT = "BitGuard-Bootstrap/1.0"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_UNSATISFIED_RANGE = re.compile(r"^bytes \*/(\d+)$")
_DIRECTORY_FSYNC_SUPPORTED = os.name != "nt"


class DownloadError(RuntimeError):
    """A source download could not be completed without risking corruption."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    destination: str
    byte_size: int
    sha256: str
    resumed: bool
    restarted: bool
    reused: bool
    source_url: str
    final_response_url: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "destination": self.destination,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "resumed": self.resumed,
            "restarted": self.restarted,
            "reused": self.reused,
            "source_url": self.source_url,
            "final_response_url": self.final_response_url,
        }


def sanitize_url(value: str) -> str:
    """Return a URL suitable for durable results and error messages."""

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise DownloadError("The source URL is invalid after removing credentials.") from error
    if not parsed.scheme or hostname is None:
        raise DownloadError("The source URL must include a scheme and host.")
    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        host = f"{host}:{port}"
    sanitized = SplitResult(parsed.scheme, host, parsed.path, parsed.query, parsed.fragment)
    return urlunsplit(sanitized)


def _validate_sha256(value: str | None) -> None:
    if value is None:
        return
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DownloadError("expected_sha256 must be an exact lowercase SHA-256 digest.")


def _resolved_destination(destination: Path | str) -> Path:
    raw = Path(destination).expanduser()
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as error:
        raise DownloadError(
            f"Download destination parent {raw.parent} must already exist: {error}."
        ) from error
    try:
        parent_stat = parent.stat()
    except OSError as error:
        raise DownloadError(f"Cannot inspect destination parent {parent}: {error}.") from error
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise DownloadError(f"Download destination parent {parent} is not a directory.")
    return parent / raw.name


def _lstat_regular(path: Path, *, allow_absent: bool) -> os.stat_result | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        if allow_absent:
            return None
        raise DownloadError(f"Required download file {path} does not exist.") from None
    except OSError as error:
        raise DownloadError(f"Cannot inspect download path {path}: {error}.") from error
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise DownloadError(f"Download path {path} must be a regular non-symlink file.")
    return value


def _hash_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise DownloadError(f"Download path {path} is no longer a regular file.")
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except DownloadError:
        raise
    except OSError as error:
        raise DownloadError(f"Cannot hash download file {path}: {error}.") from error
    return size, digest.hexdigest()


def _parse_content_length(headers: Any) -> int | None:
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise DownloadError("The HTTP response has an invalid Content-Length.") from error
    if value < 0:
        raise DownloadError("The HTTP response has a negative Content-Length.")
    return value


def _validated_partial_range(
    headers: Any, expected_start: int
) -> tuple[int, int]:
    raw = headers.get("Content-Range")
    match = _CONTENT_RANGE.fullmatch(raw or "")
    if match is None:
        raise DownloadError("The resumed response has an invalid Content-Range.")
    start, end, total = (int(value) for value in match.groups())
    if start != expected_start or end < start or total <= end:
        raise DownloadError(
            "The resumed response Content-Range does not match the local partial file."
        )
    content_length = _parse_content_length(headers)
    range_length = end - start + 1
    if content_length is None or content_length != range_length:
        raise DownloadError(
            "The resumed response Content-Range and Content-Length are inconsistent."
        )
    return total, content_length


def _write_all(stream: Any, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = stream.write(payload[offset:])
        if written is None or written <= 0:
            raise OSError("partial-file write made no progress")
        offset += written


def _fsync_parent_directory(path: Path) -> None:
    if not _DIRECTORY_FSYNC_SUPPORTED:
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    failure: OSError | None = None
    try:
        os.fsync(descriptor)
    except OSError as error:
        failure = error
    try:
        os.close(descriptor)
    except OSError as close_error:
        if failure is not None:
            raise OSError(
                f"directory fsync failed: {failure}; close also failed: {close_error}"
            ) from failure
        raise
    if failure is not None:
        raise failure


def _open_partial(path: Path, *, append: bool, known_stat: os.stat_result | None):
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    if append:
        flags |= os.O_APPEND
    else:
        flags |= os.O_TRUNC
    if known_stat is None:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise DownloadError(f"Partial download {path} is not a regular file.")
        if known_stat is not None and (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ) != (known_stat.st_dev, known_stat.st_ino):
            raise DownloadError(f"Partial download {path} changed before it was opened.")
        return os.fdopen(descriptor, "ab" if append else "wb", buffering=0)
    except DownloadError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise
    except OSError as error:
        raise DownloadError(f"Cannot safely open partial download {path}: {error}.") from error


def _stream_response(
    response: Any,
    partial: Path,
    *,
    append: bool,
    known_stat: os.stat_result | None,
    chunk_size: int,
) -> int:
    received = 0
    stream = _open_partial(partial, append=append, known_stat=known_stat)
    failure: BaseException | None = None
    try:
        while True:
            try:
                chunk = response.read(chunk_size)
            except http.client.IncompleteRead as error:
                if error.partial:
                    _write_all(stream, error.partial)
                    received += len(error.partial)
                raise
            if not chunk:
                break
            _write_all(stream, chunk)
            received += len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    except Exception as error:
        failure = error
        try:
            stream.flush()
            os.fsync(stream.fileno())
        except OSError as sync_error:
            failure = OSError(f"{error}; preserving the partial also failed: {sync_error}")
    try:
        stream.close()
    except OSError as close_error:
        if failure is None:
            failure = close_error
        else:
            failure = OSError(f"{failure}; partial close also failed: {close_error}")
    if failure is not None:
        if isinstance(failure, DownloadError):
            raise failure
        raise DownloadError(
            f"The download response ended unsuccessfully; resumable content remains at "
            f"{partial}: {failure}."
        ) from failure
    return received


def _publish_partial(partial: Path, destination: Path) -> None:
    partial_stat = _lstat_regular(partial, allow_absent=False)
    assert partial_stat is not None
    try:
        os.link(partial, destination)
    except FileExistsError as error:
        raise DownloadError(
            f"Download destination {destination} already exists, possibly from a concurrent "
            f"publisher; the verified partial remains at {partial}."
        ) from error
    except OSError as error:
        raise DownloadError(
            f"Cannot atomically publish verified partial {partial} without overwriting "
            f"{destination}: {error}."
        ) from error
    try:
        _fsync_parent_directory(destination)
    except OSError as error:
        raise DownloadError(
            f"Download {destination} was published but its directory could not be made "
            f"durable: {error}; inspect the final and partial before retrying."
        ) from error

    try:
        current = partial.lstat()
        final = destination.lstat()
        identity = (partial_stat.st_dev, partial_stat.st_ino)
        if (current.st_dev, current.st_ino) != identity or (
            final.st_dev,
            final.st_ino,
        ) != identity:
            raise DownloadError(
                f"Download {destination} was published, but owned partial {partial} changed; "
                "it was preserved for inspection."
            )
        partial.unlink()
        _fsync_parent_directory(destination)
    except DownloadError:
        raise
    except OSError as error:
        raise DownloadError(
            f"Download {destination} was published, but exact partial cleanup failed: {error}."
        ) from error


def download_file(
    source_url: str,
    destination: Path | str,
    *,
    expected_sha256: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DownloadResult:
    """Download one URL through a sole resumable ``.partial`` path."""

    safe_source = sanitize_url(source_url)
    _validate_sha256(expected_sha256)
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= MAX_TIMEOUT_SECONDS:
        raise DownloadError(
            f"timeout must be greater than zero and at most {MAX_TIMEOUT_SECONDS} seconds."
        )
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise DownloadError("chunk_size must be a positive integer.")

    target = _resolved_destination(destination)
    partial = target.with_name(f"{target.name}.partial")
    final_stat = _lstat_regular(target, allow_absent=True)
    if final_stat is not None:
        if expected_sha256 is None:
            raise DownloadError(
                f"Download destination {target} already exists; provide a persisted expected "
                "hash before trusting or reusing it."
            )
        size, digest = _hash_file(target, chunk_size)
        if digest != expected_sha256:
            raise DownloadError(
                f"Existing download {target} does not match the expected SHA-256 and will "
                "not be replaced."
            )
        return DownloadResult(
            destination=str(target),
            byte_size=size,
            sha256=digest,
            resumed=False,
            restarted=False,
            reused=True,
            source_url=safe_source,
            final_response_url=None,
        )

    partial_stat = _lstat_regular(partial, allow_absent=True)
    offset = partial_stat.st_size if partial_stat is not None else 0
    retry_without_range = False
    restarted = False
    response_url: str | None = None
    expected_total: int | None = None
    resumed = False

    while True:
        request_offset = 0 if retry_without_range else offset
        headers = {"User-Agent": USER_AGENT}
        if request_offset > 0:
            headers["Range"] = f"bytes={request_offset}-"
        request = Request(source_url, headers=headers, method="GET")
        try:
            response = urlopen(request, timeout=float(timeout))
        except HTTPError as error:
            if error.code == 416 and request_offset > 0 and not retry_without_range:
                response_url = sanitize_url(error.geturl() or safe_source)
                match = _UNSATISFIED_RANGE.fullmatch(
                    error.headers.get("Content-Range", "") if error.headers else ""
                )
                if match is not None and int(match.group(1)) == request_offset:
                    expected_total = request_offset
                    break
                retry_without_range = True
                restarted = True
                continue
            raise DownloadError(
                f"HTTP download failed for {safe_source} with status {error.code}; "
                f"resumable content remains at {partial}."
            ) from None
        except (URLError, TimeoutError, OSError, ValueError) as error:
            raise DownloadError(
                f"Network download failed for {safe_source}; resumable content remains at "
                f"{partial}: {type(error).__name__}."
            ) from None

        try:
            with response:
                response_url = sanitize_url(response.geturl() or safe_source)
                status_code = getattr(response, "status", response.getcode())
                if request_offset > 0:
                    if status_code == 206:
                        expected_total, expected_body = _validated_partial_range(
                            response.headers, request_offset
                        )
                        append = True
                        resumed = True
                    elif status_code == 200:
                        expected_body = _parse_content_length(response.headers)
                        expected_total = expected_body
                        append = False
                        restarted = True
                        resumed = False
                    else:
                        raise DownloadError(
                            f"Unexpected HTTP status {status_code} while resuming {safe_source}."
                        )
                else:
                    if status_code != 200:
                        raise DownloadError(
                            f"Unexpected HTTP status {status_code} while downloading {safe_source}."
                        )
                    expected_body = _parse_content_length(response.headers)
                    expected_total = expected_body
                    append = False
                received = _stream_response(
                    response,
                    partial,
                    append=append,
                    known_stat=partial_stat,
                    chunk_size=chunk_size,
                )
                if expected_body is not None and received != expected_body:
                    raise DownloadError(
                        f"HTTP response length mismatch for {safe_source}: expected "
                        f"{expected_body} bytes but received {received}; resumable content "
                        f"remains at {partial}."
                    )
            break
        except DownloadError:
            raise
        except (http.client.HTTPException, OSError, ValueError) as error:
            raise DownloadError(
                f"HTTP response processing failed for {safe_source}; resumable content "
                f"remains at {partial}: {type(error).__name__}."
            ) from error

    partial_stat = _lstat_regular(partial, allow_absent=False)
    assert partial_stat is not None
    if expected_total is not None and partial_stat.st_size != expected_total:
        raise DownloadError(
            f"Complete partial size mismatch for {safe_source}: expected {expected_total} bytes "
            f"but have {partial_stat.st_size}; the partial was not published."
        )
    size, digest = _hash_file(partial, chunk_size)
    if expected_sha256 is not None and digest != expected_sha256:
        raise DownloadError(
            f"Downloaded content SHA-256 does not match the expected SHA-256; verified "
            f"publication was refused and {partial} remains resumable."
        )
    _publish_partial(partial, target)
    return DownloadResult(
        destination=str(target),
        byte_size=size,
        sha256=digest,
        resumed=resumed,
        restarted=restarted,
        reused=False,
        source_url=safe_source,
        final_response_url=response_url,
    )
