"""Crash-safe, resumable HTTP downloads for bootstrap source archives."""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import stat
import uuid
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


@dataclass(frozen=True, slots=True)
class _StreamResult:
    received: int
    identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _VerifiedSnapshot:
    pin: Path
    identity: tuple[int, int]
    source_identity: tuple[int, int]
    byte_size: int
    sha256: str


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
    sanitized = SplitResult(parsed.scheme, host, parsed.path, "", "")
    return urlunsplit(sanitized)


def _normalize_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise DownloadError("expected_sha256 must be a string.")
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DownloadError("expected_sha256 must be an exact lowercase SHA-256 digest.")
    return value


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


def _file_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _content_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_path_identity(
    path: Path, expected: tuple[int, int], *, subject: str
) -> os.stat_result:
    current = _lstat_regular(path, allow_absent=False)
    assert current is not None
    if _file_identity(current) != expected:
        raise DownloadError(
            f"{subject} {path} changed identity; refusing to hash or publish another file."
        )
    return current


def _hash_file(
    path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> tuple[int, str, tuple[int, int]]:
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            opened_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise DownloadError(f"Download path {path} is no longer a regular file.")
            opened_identity = _file_identity(opened_stat)
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
            if _file_identity(os.fstat(stream.fileno())) != opened_identity:
                raise DownloadError(f"Download path {path} changed identity while hashing.")
    except DownloadError:
        raise
    except OSError as error:
        raise DownloadError(f"Cannot hash download file {path}: {error}.") from error
    return size, digest.hexdigest(), opened_identity


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
    if known_stat is None:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise DownloadError(f"Partial download {path} is not a regular file.")
        if known_stat is not None and _file_identity(opened_stat) != _file_identity(
            known_stat
        ):
            raise DownloadError(f"Partial download {path} changed before it was opened.")
        if not append:
            os.ftruncate(descriptor, 0)
        identity = _file_identity(opened_stat)
        stream = os.fdopen(descriptor, "ab" if append else "wb", buffering=0)
        descriptor = None
        return stream, identity
    except DownloadError:
        raise
    except OSError as error:
        raise DownloadError(f"Cannot safely open partial download {path}: {error}.") from error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _stream_response(
    response: Any,
    partial: Path,
    *,
    append: bool,
    known_stat: os.stat_result | None,
    chunk_size: int,
) -> _StreamResult:
    received = 0
    stream, opened_identity = _open_partial(
        partial, append=append, known_stat=known_stat
    )
    failure: Exception | None = None
    base_exception_active = False
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
        if _file_identity(os.fstat(stream.fileno())) != opened_identity:
            raise DownloadError(
                f"Partial download {partial} changed identity while streaming."
            )
    except Exception as error:
        failure = error
        try:
            stream.flush()
            os.fsync(stream.fileno())
        except OSError as sync_error:
            failure = OSError(f"{error}; preserving the partial also failed: {sync_error}")
    except BaseException:
        base_exception_active = True
        try:
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            pass
        raise
    finally:
        try:
            stream.close()
        except OSError as close_error:
            if not base_exception_active:
                if failure is None:
                    failure = close_error
                else:
                    failure = OSError(
                        f"{failure}; partial close also failed: {close_error}"
                    )
    if failure is not None:
        if isinstance(failure, DownloadError):
            raise failure
        raise DownloadError(
            f"The download response ended unsuccessfully; resumable content remains at "
            f"{partial}: {failure}."
        ) from failure
    return _StreamResult(received=received, identity=opened_identity)


def _unlink_private_candidate(candidate: Path, expected: tuple[int, int]) -> None:
    try:
        current = candidate.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise DownloadError(f"Cannot inspect private publish link {candidate}: {error}.") from error
    if _file_identity(current) != expected:
        raise DownloadError(
            f"Private publish link {candidate} changed identity and was preserved."
        )
    try:
        candidate.unlink()
    except OSError as error:
        raise DownloadError(f"Cannot remove private publish link {candidate}: {error}.") from error


def _create_private_snapshot(partial: Path) -> tuple[Path, int]:
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    for _attempt in range(16):
        candidate = partial.with_name(
            f".{partial.name}.{uuid.uuid4().hex}.verified"
        )
        try:
            return candidate, os.open(candidate, flags, 0o600)
        except FileExistsError:
            continue
        except OSError as error:
            raise DownloadError(
                f"Cannot create a private verified snapshot for {partial}: {error}."
            ) from error
    raise DownloadError(f"Cannot reserve a private verified snapshot for {partial}.")


def _write_descriptor(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("verified snapshot write made no progress")
        offset += written


def _pin_private_snapshot(
    private: Path,
    descriptor: int,
    expected_identity: tuple[int, int],
) -> Path:
    for _attempt in range(16):
        pin = private.with_name(f".{private.name}.{uuid.uuid4().hex}.verified.pin")
        try:
            os.link(private, pin)
        except FileExistsError:
            continue
        except OSError as error:
            raise DownloadError(
                f"Cannot pin private verified snapshot {private}: {error}."
            ) from error
        pin_stat = _lstat_regular(pin, allow_absent=False)
        assert pin_stat is not None
        pin_identity = _file_identity(pin_stat)
        descriptor_identity = _file_identity(os.fstat(descriptor))
        try:
            private_stat = _lstat_regular(private, allow_absent=False)
            assert private_stat is not None
            private_identity = _file_identity(private_stat)
        except DownloadError:
            if pin_identity == expected_identity:
                _unlink_private_candidate(pin, expected_identity)
            raise
        if (
            pin_identity != expected_identity
            or descriptor_identity != expected_identity
            or private_identity != expected_identity
        ):
            if pin_identity == expected_identity:
                _unlink_private_candidate(pin, expected_identity)
            raise DownloadError(
                f"Private verified snapshot {private} changed identity while it was pinned."
            )
        return pin
    raise DownloadError(f"Cannot reserve a private publish pin for {private}.")


def _snapshot_partial(
    partial: Path,
    expected_identity: tuple[int, int],
    chunk_size: int,
) -> _VerifiedSnapshot:
    _require_path_identity(partial, expected_identity, subject="Completed partial")
    source_descriptor: int | None = None
    private_descriptor: int | None = None
    private: Path | None = None
    private_identity: tuple[int, int] | None = None
    pin: Path | None = None
    success = False
    base_exception_active = False
    size = 0
    digest = hashlib.sha256()
    try:
        try:
            source_flags = (
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_BINARY", 0)
            )
            source_descriptor = os.open(partial, source_flags)
            source_before = os.fstat(source_descriptor)
            if not stat.S_ISREG(source_before.st_mode):
                raise DownloadError(f"Completed partial {partial} is not a regular file.")
            if _file_identity(source_before) != expected_identity:
                raise DownloadError(
                    f"Completed partial {partial} changed before snapshot verification."
                )
            source_version = _content_identity(source_before)

            private, private_descriptor = _create_private_snapshot(partial)
            private_before = os.fstat(private_descriptor)
            if not stat.S_ISREG(private_before.st_mode):
                raise DownloadError(
                    f"Private verified snapshot {private} is not a regular file."
                )
            private_identity = _file_identity(private_before)

            while True:
                chunk = os.read(source_descriptor, chunk_size)
                if not chunk:
                    break
                _write_descriptor(private_descriptor, chunk)
                digest.update(chunk)
                size += len(chunk)
            os.fsync(private_descriptor)

            if _content_identity(os.fstat(source_descriptor)) != source_version:
                raise DownloadError(
                    f"Completed partial {partial} changed while its verified snapshot was copied."
                )
            private_after = os.fstat(private_descriptor)
            if (
                _file_identity(private_after) != private_identity
                or private_after.st_size != size
            ):
                raise DownloadError(
                    f"Private verified snapshot {private} changed while it was written."
                )
            _require_path_identity(partial, expected_identity, subject="Hashed partial")
            pin = _pin_private_snapshot(
                private,
                private_descriptor,
                private_identity,
            )
        except OSError as error:
            raise DownloadError(
                f"Cannot create a verified snapshot of partial download {partial}: {error}."
            ) from error
        except BaseException:
            base_exception_active = True
            raise
        finally:
            close_errors: list[OSError] = []
            for descriptor in (source_descriptor, private_descriptor):
                if descriptor is None:
                    continue
                try:
                    os.close(descriptor)
                except OSError as close_error:
                    close_errors.append(close_error)
            if close_errors and not base_exception_active:
                raise DownloadError(
                    f"Could not close verified snapshot descriptors for {partial}: "
                    f"{close_errors[0]}."
                ) from close_errors[0]

        assert private is not None
        assert private_identity is not None
        assert pin is not None
        _unlink_private_candidate(private, private_identity)
        success = True
        return _VerifiedSnapshot(
            pin=pin,
            identity=private_identity,
            source_identity=expected_identity,
            byte_size=size,
            sha256=digest.hexdigest(),
        )
    finally:
        if not success:
            for candidate, identity in (
                (pin, private_identity),
                (private, private_identity),
            ):
                if candidate is None or identity is None:
                    continue
                try:
                    _unlink_private_candidate(candidate, identity)
                except Exception:
                    pass


def _publish_snapshot(
    snapshot: _VerifiedSnapshot,
    partial: Path,
    destination: Path,
) -> None:
    pin_stat = _lstat_regular(snapshot.pin, allow_absent=False)
    assert pin_stat is not None
    if _file_identity(pin_stat) != snapshot.identity:
        raise DownloadError(
            f"Private verified snapshot pin {snapshot.pin} changed before publication."
        )
    try:
        os.link(snapshot.pin, destination)
    except FileExistsError as error:
        _unlink_private_candidate(snapshot.pin, snapshot.identity)
        raise DownloadError(
            f"Download destination {destination} already exists, possibly from a concurrent "
            f"publisher; the verified partial remains at {partial}."
        ) from error
    except OSError as error:
        _unlink_private_candidate(snapshot.pin, snapshot.identity)
        raise DownloadError(
            f"Cannot atomically publish verified snapshot of {partial} without overwriting "
            f"{destination}: {error}."
        ) from error
    final_stat = _lstat_regular(destination, allow_absent=False)
    assert final_stat is not None
    final_identity = _file_identity(final_stat)
    if final_identity != snapshot.identity:
        raise DownloadError(
            f"Download destination {destination} was not linked from the verified identity; "
            "the unowned final path was preserved and the private verified pin will be "
            "cleaned if it still has its known identity."
        )
    try:
        _fsync_parent_directory(destination)
    except OSError as error:
        raise DownloadError(
            f"Download {destination} was published but its directory could not be made "
            f"durable: {error}; inspect the final and partial before retrying."
        ) from error

    try:
        _unlink_private_candidate(snapshot.pin, snapshot.identity)
        _require_path_identity(partial, snapshot.source_identity, subject="Owned partial")
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
    expected_sha256 = (
        None if expected_sha256 is None else _normalize_sha256(expected_sha256)
    )
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= MAX_TIMEOUT_SECONDS
    ):
        raise DownloadError(
            f"timeout must be greater than zero and at most {MAX_TIMEOUT_SECONDS} seconds."
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
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
        size, digest, hashed_identity = _hash_file(target, chunk_size)
        if hashed_identity != _file_identity(final_stat):
            raise DownloadError(
                f"Existing download {target} changed identity before it could be reused."
            )
        _require_path_identity(target, hashed_identity, subject="Existing download")
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
    verified_identity = _file_identity(partial_stat) if partial_stat is not None else None

    while True:
        request_offset = 0 if retry_without_range else offset
        headers = {"User-Agent": USER_AGENT}
        if request_offset > 0:
            headers["Range"] = f"bytes={request_offset}-"
        request = Request(source_url, headers=headers, method="GET")
        try:
            response = urlopen(request, timeout=float(timeout))
        except HTTPError as error:
            try:
                status_code = error.code
                error_response_url = sanitize_url(error.geturl() or safe_source)
                content_range = (
                    error.headers.get("Content-Range", "") if error.headers else ""
                )
            finally:
                try:
                    error.close()
                except OSError as close_error:
                    raise DownloadError(
                        f"Could not close HTTP error response for {safe_source}: "
                        f"{close_error}."
                    ) from close_error
            if status_code == 416 and request_offset > 0 and not retry_without_range:
                response_url = error_response_url
                match = _UNSATISFIED_RANGE.fullmatch(content_range)
                if match is not None and int(match.group(1)) == request_offset:
                    expected_total = request_offset
                    break
                retry_without_range = True
                restarted = True
                continue
            raise DownloadError(
                f"HTTP download failed for {error_response_url} with status {status_code}; "
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
                stream_result = _stream_response(
                    response,
                    partial,
                    append=append,
                    known_stat=partial_stat,
                    chunk_size=chunk_size,
                )
                verified_identity = stream_result.identity
                if expected_body is not None and stream_result.received != expected_body:
                    raise DownloadError(
                        f"HTTP response length mismatch for {safe_source}: expected "
                        f"{expected_body} bytes but received {stream_result.received}; "
                        "resumable content "
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

    if verified_identity is None:
        raise DownloadError(f"Partial download {partial} has no verified file identity.")
    partial_stat = _require_path_identity(
        partial, verified_identity, subject="Completed partial"
    )
    snapshot = _snapshot_partial(partial, verified_identity, chunk_size)
    published = False
    try:
        if expected_total is not None and snapshot.byte_size != expected_total:
            raise DownloadError(
                f"Complete partial size mismatch for {safe_source}: expected "
                f"{expected_total} bytes but copied {snapshot.byte_size}; the partial was "
                "not published."
            )
        if partial_stat.st_size != snapshot.byte_size:
            raise DownloadError(
                f"Completed partial {partial} changed size before its verified snapshot "
                "could be published."
            )
        if expected_sha256 is not None and snapshot.sha256 != expected_sha256:
            raise DownloadError(
                f"Downloaded content SHA-256 does not match the expected SHA-256; verified "
                f"publication was refused and {partial} remains resumable."
            )
        _publish_snapshot(snapshot, partial, target)
        published = True
    finally:
        if not published:
            try:
                _unlink_private_candidate(snapshot.pin, snapshot.identity)
            except Exception:
                pass
    return DownloadResult(
        destination=str(target),
        byte_size=snapshot.byte_size,
        sha256=snapshot.sha256,
        resumed=resumed,
        restarted=restarted,
        reused=False,
        source_url=safe_source,
        final_response_url=response_url,
    )
