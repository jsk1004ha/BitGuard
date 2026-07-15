"""Crash-safe bootstrap stage state and exclusive writer ownership."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import socket
import stat
import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


STATE_FORMAT_VERSION = 1
LOCK_FORMAT_VERSION = 1
STAGE_ORDER = (
    "preflight",
    "environment",
    "acquire",
    "extract",
    "inspect",
    "shard",
    "validate",
    "train",
    "summarize",
)
_DIRECTORY_FSYNC_SUPPORTED = os.name != "nt"


class BootstrapStateError(RuntimeError):
    """The persisted bootstrap state is unsafe or cannot be updated."""


class BootstrapLockError(RuntimeError):
    """Exclusive bootstrap writer ownership cannot be established safely."""


def _exclusive_write_flags() -> int:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    return flags | getattr(os, "O_NOFOLLOW", 0)


def _write_all(descriptor: int, payload: bytes, subject: str) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError(f"{subject} write made no progress")
        offset += written


def _fsync_parent_directory(path: Path) -> None:
    """Durably commit a replaced directory entry where the platform permits it."""

    # CPython on Windows cannot open directories through os.open, so there is no
    # standard-library directory descriptor that can be passed to os.fsync.
    if not _DIRECTORY_FSYNC_SUPPORTED:
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    sync_error: OSError | None = None
    try:
        os.fsync(descriptor)
    except OSError as error:
        sync_error = error
    try:
        os.close(descriptor)
    except OSError as close_error:
        if sync_error is not None:
            raise OSError(
                f"directory fsync failed: {sync_error}; directory close failed: "
                f"{close_error}"
            ) from sync_error
        raise
    if sync_error is not None:
        raise sync_error


class BootstrapStateStore:
    """Persist completed bootstrap stages with portable output fingerprints."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.root = self.path.parent
        self._stages = self._load()

    @property
    def completed_stages(self) -> tuple[str, ...]:
        """Return completed stages in canonical pipeline order."""

        return tuple(stage for stage in STAGE_ORDER if stage in self._stages)

    def complete(
        self,
        stage: str,
        input_signature: str,
        outputs: Iterable[Path | str],
    ) -> None:
        """Record a stage after fingerprinting all of its regular-file outputs."""

        self._require_stage(stage)
        self._require_signature(input_signature)
        fingerprints = [
            self._validate_stored_fingerprint(stage, self._fingerprint(Path(output)))
            for output in outputs
        ]
        fingerprints.sort(key=lambda item: item["path"])
        paths = [item["path"] for item in fingerprints]
        if len(paths) != len(set(paths)):
            raise BootstrapStateError(
                f"Stage {stage!r} lists the same output path more than once."
            )

        updated = dict(self._stages)
        updated[stage] = {
            "input_signature": input_signature,
            "outputs": fingerprints,
        }
        self._persist(updated)
        self._stages = updated

    def reusable(self, stage: str, input_signature: str) -> bool:
        """Return whether a completed stage has identical inputs and outputs."""

        self._require_stage(stage)
        self._require_signature(input_signature)
        record = self._stages.get(stage)
        if record is None or record["input_signature"] != input_signature:
            return False

        for expected in record["outputs"]:
            relative = PurePosixPath(expected["path"])
            output = self.root.joinpath(*relative.parts)
            try:
                actual = self._fingerprint(output)
            except (BootstrapStateError, OSError):
                return False
            if actual != expected:
                return False
        return True

    def invalidate_from(self, stage: str, order: Sequence[str]) -> None:
        """Remove a stage and all dependants according to the canonical order."""

        self._require_stage(stage)
        supplied_order = tuple(order)
        if supplied_order != STAGE_ORDER:
            raise BootstrapStateError(
                "Restart invalidation requires the canonical stage order: "
                + " -> ".join(STAGE_ORDER)
            )

        restart_index = supplied_order.index(stage)
        invalidated = set(supplied_order[restart_index:])
        updated = {
            name: record
            for name, record in self._stages.items()
            if name not in invalidated
        }
        self._persist(updated)
        self._stages = updated

    @staticmethod
    def _require_stage(stage: str) -> None:
        if stage not in STAGE_ORDER:
            raise BootstrapStateError(
                f"unknown stage {stage!r}; expected one of: {', '.join(STAGE_ORDER)}."
            )

    @staticmethod
    def _require_signature(input_signature: str) -> None:
        if not isinstance(input_signature, str):
            raise BootstrapStateError("A stage input signature must be a string.")

    def _fingerprint(self, output: Path) -> dict[str, Any]:
        candidate = output.expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, ValueError) as error:
            raise BootstrapStateError(
                f"Bootstrap output {candidate} must exist and be a regular file."
            ) from error

        try:
            mode = resolved.stat().st_mode
        except OSError as error:
            raise BootstrapStateError(
                f"Cannot inspect bootstrap output {resolved}: {error}."
            ) from error
        if not stat.S_ISREG(mode):
            raise BootstrapStateError(
                f"Bootstrap output {resolved} must exist and be a regular file."
            )

        try:
            relative = resolved.relative_to(self.root)
        except ValueError as error:
            raise BootstrapStateError(
                f"Bootstrap output {resolved} is outside the state root {self.root}; "
                "only portable relative output paths can be recorded."
            ) from error

        digest = hashlib.sha256()
        try:
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            size = resolved.stat().st_size
        except OSError as error:
            raise BootstrapStateError(
                f"Cannot fingerprint bootstrap output {resolved}: {error}."
            ) from error

        return {
            "path": relative.as_posix(),
            "size": size,
            "sha256": digest.hexdigest(),
        }

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as error:
            raise BootstrapStateError(
                f"Cannot read bootstrap state {self.path}: {error}."
            ) from error
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise BootstrapStateError(
                f"Bootstrap state {self.path} is malformed; restore or remove it "
                "before restarting the bootstrap."
            ) from error

        return self._validate_payload(payload)

    def _validate_payload(self, payload: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(payload, dict) or set(payload) != {"version", "stages"}:
            raise BootstrapStateError(
                f"Bootstrap state {self.path} has an invalid state document shape."
            )
        version = payload["version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise BootstrapStateError(
                f"Bootstrap state {self.path} has an invalid state format version."
            )
        if version > STATE_FORMAT_VERSION:
            raise BootstrapStateError(
                f"Bootstrap state {self.path} uses newer format version {version}; "
                "upgrade BitGuard before resuming."
            )
        if version != STATE_FORMAT_VERSION:
            raise BootstrapStateError(
                f"Bootstrap state {self.path} uses unsupported format version "
                f"{version}; recreate it with this BitGuard version."
            )

        stages = payload["stages"]
        if not isinstance(stages, dict):
            raise BootstrapStateError(
                f"Bootstrap state {self.path} has an invalid stages mapping."
            )
        validated: dict[str, dict[str, Any]] = {}
        for stage, record in stages.items():
            if stage not in STAGE_ORDER:
                raise BootstrapStateError(
                    f"Bootstrap state {self.path} contains unknown stage {stage!r}."
                )
            if not isinstance(record, dict) or set(record) != {
                "input_signature",
                "outputs",
            }:
                raise BootstrapStateError(
                    f"Bootstrap state stage {stage!r} has an invalid state record."
                )
            self._require_signature(record["input_signature"])
            outputs = record["outputs"]
            if not isinstance(outputs, list):
                raise BootstrapStateError(
                    f"Bootstrap state stage {stage!r} has invalid state outputs."
                )
            validated_outputs = [
                self._validate_stored_fingerprint(stage, fingerprint)
                for fingerprint in outputs
            ]
            paths = [fingerprint["path"] for fingerprint in validated_outputs]
            if paths != sorted(paths) or len(paths) != len(set(paths)):
                raise BootstrapStateError(
                    f"Bootstrap state stage {stage!r} has non-deterministic or "
                    "duplicate output paths."
                )
            validated[stage] = {
                "input_signature": record["input_signature"],
                "outputs": validated_outputs,
            }
        return validated

    def _validate_stored_fingerprint(
        self, stage: str, fingerprint: Any
    ) -> dict[str, Any]:
        if not isinstance(fingerprint, dict) or set(fingerprint) != {
            "path",
            "size",
            "sha256",
        }:
            raise BootstrapStateError(
                f"Bootstrap state stage {stage!r} has an invalid output fingerprint."
            )

        path = fingerprint["path"]
        if not isinstance(path, str) or not path:
            raise BootstrapStateError(
                f"Bootstrap state stage {stage!r} has an invalid output path."
            )
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or path != relative.as_posix()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in path
            or ":" in path
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in path)
        ):
            raise BootstrapStateError(
                f"Bootstrap state stage {stage!r} has an unsafe relative state path."
            )

        size = fingerprint["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BootstrapStateError(
                f"Bootstrap state stage {stage!r} has an invalid output size."
            )
        sha256 = fingerprint["sha256"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise BootstrapStateError(
                f"Bootstrap state stage {stage!r} has an invalid SHA-256 digest."
            )
        return {"path": path, "size": size, "sha256": sha256}

    def _persist(self, stages: dict[str, dict[str, Any]]) -> None:
        payload = {"version": STATE_FORMAT_VERSION, "stages": stages}
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        encoded = serialized.encode("utf-8")
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        try:
            descriptor = os.open(temporary, _exclusive_write_flags(), 0o600)
        except FileExistsError as error:
            raise BootstrapStateError(
                f"Bootstrap state temporary file {temporary} already exists; it "
                "was not modified or removed; inspect the competing or interrupted "
                "writer and remove the file only when safe."
            ) from error
        except OSError as error:
            raise BootstrapStateError(
                f"Could not create bootstrap state temporary file {temporary}: "
                f"{error}."
            ) from error

        identity: tuple[int, int] | None = None
        operation = "identity"
        failure: OSError | None = None
        try:
            identity = _stat_identity(os.fstat(descriptor))
            operation = "write"
            _write_all(descriptor, encoded, "state temporary file")
            operation = "fsync"
            os.fsync(descriptor)
        except OSError as error:
            failure = error
        try:
            os.close(descriptor)
        except OSError as error:
            if failure is None:
                operation = "close"
                failure = error
            else:
                operation = f"{operation}/close"
                failure = OSError(f"{failure}; close also failed: {error}")

        if failure is not None:
            if identity is None:
                raise BootstrapStateError(
                    f"Could not establish the identity of created bootstrap state "
                    f"temporary file {temporary}: {failure}; the file is preserved "
                    "for inspection because this writer cannot prove it still owns "
                    "the path."
                ) from failure
            cleanup_error: BootstrapStateError | None = None
            try:
                self._discard_owned_temporary(temporary, identity)
            except BootstrapStateError as error:
                cleanup_error = error
            detail = (
                f"; safe temporary-file cleanup also failed: {cleanup_error}"
                if cleanup_error is not None
                else ""
            )
            raise BootstrapStateError(
                f"Could not persist bootstrap state {self.path} during {operation}: "
                f"{failure}{detail}."
            ) from failure

        try:
            temporary.replace(self.path)
        except OSError as error:
            cleanup_error: BootstrapStateError | None = None
            try:
                self._discard_owned_temporary(temporary, identity)
            except BootstrapStateError as caught:
                cleanup_error = caught
            detail = (
                f"; safe temporary-file cleanup also failed: {cleanup_error}"
                if cleanup_error is not None
                else ""
            )
            raise BootstrapStateError(
                f"Could not persist bootstrap state {self.path} during replace: "
                f"{error}{detail}."
            ) from error

        try:
            _fsync_parent_directory(self.path)
        except OSError as error:
            raise BootstrapStateError(
                f"Bootstrap state {self.path} was replaced, but its parent "
                f"directory could not be made durable: {error}. The new state may "
                "already be visible; inspect it before retrying."
            ) from error

    def _discard_owned_temporary(
        self, temporary: Path, expected_identity: tuple[int, int]
    ) -> None:
        quarantine = self._reserve_state_quarantine(temporary)
        try:
            os.replace(temporary, quarantine)
        except FileNotFoundError:
            try:
                quarantine.unlink()
            except OSError as error:
                raise BootstrapStateError(
                    f"Created state temporary {temporary} disappeared, and reserved "
                    f"quarantine {quarantine} could not be cleaned: {error}."
                ) from error
            return
        except OSError as error:
            try:
                quarantine.unlink()
            except OSError as cleanup_error:
                raise BootstrapStateError(
                    f"Cannot quarantine state temporary {temporary}: {error}; "
                    f"reserved quarantine {quarantine} also could not be cleaned: "
                    f"{cleanup_error}."
                ) from error
            raise BootstrapStateError(
                f"Cannot quarantine state temporary {temporary}: {error}."
            ) from error

        try:
            quarantined_identity = _stat_identity(
                quarantine.stat(follow_symlinks=False)
            )
        except OSError as error:
            self._restore_state_quarantine(
                quarantine, temporary, "unverifiable temporary cleanup"
            )
            raise BootstrapStateError(
                f"Cannot verify quarantined state temporary {quarantine}: {error}; "
                "it was restored without deletion."
            ) from error

        if quarantined_identity != expected_identity:
            self._restore_state_quarantine(
                quarantine, temporary, "changed temporary cleanup candidate"
            )
            return

        try:
            quarantine.unlink()
        except OSError as error:
            self._restore_state_quarantine(
                quarantine, temporary, "failed temporary cleanup"
            )
            raise BootstrapStateError(
                f"Cannot clean created state temporary {temporary}: {error}; it "
                "was restored without deletion."
            ) from error

    @staticmethod
    def _reserve_state_quarantine(temporary: Path) -> Path:
        for _attempt in range(16):
            quarantine = temporary.with_name(
                f".{temporary.name}.{uuid.uuid4().hex}.quarantine"
            )
            try:
                descriptor = os.open(quarantine, _exclusive_write_flags(), 0o600)
            except FileExistsError:
                continue
            except OSError as error:
                raise BootstrapStateError(
                    f"Cannot reserve a quarantine for state temporary {temporary}: "
                    f"{error}."
                ) from error
            try:
                os.close(descriptor)
            except OSError as error:
                raise BootstrapStateError(
                    f"Cannot close reserved state quarantine {quarantine}: {error}; "
                    "inspect and remove it only when safe."
                ) from error
            return quarantine
        raise BootstrapStateError(
            f"Cannot reserve a unique quarantine for state temporary {temporary}; "
            "inspect sibling quarantine files before retrying."
        )

    @staticmethod
    def _restore_state_quarantine(
        quarantine: Path, temporary: Path, reason: str
    ) -> None:
        try:
            os.link(quarantine, temporary, follow_symlinks=False)
        except OSError as error:
            raise BootstrapStateError(
                f"Cannot safely restore state temporary {temporary} after {reason}: "
                f"{error}; quarantine {quarantine} is preserved for inspection."
            ) from error
        try:
            quarantine.unlink()
        except OSError as error:
            raise BootstrapStateError(
                f"State temporary {temporary} was restored after {reason}, but "
                f"quarantine {quarantine} could not be removed: {error}."
            ) from error


class BootstrapWriterLock:
    """A fail-closed, context-manageable single-writer bootstrap lock."""

    def __init__(
        self,
        path: Path | str,
        *,
        recover_stale: bool = False,
        pid: int | None = None,
        hostname: str | None = None,
        clock: Callable[[], datetime] | None = None,
        nonce_factory: Callable[[], str] | None = None,
        pid_is_alive: Callable[[int], bool] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.recover_stale = recover_stale
        self.pid = os.getpid() if pid is None else pid
        self.hostname = socket.gethostname() if hostname is None else hostname
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._nonce_factory = nonce_factory or (lambda: uuid.uuid4().hex)
        self._pid_is_alive = pid_is_alive or _pid_is_alive
        self._owned_metadata: dict[str, Any] | None = None
        self._owned_identity: tuple[int, int] | None = None
        if not isinstance(self.pid, int) or isinstance(self.pid, bool) or self.pid <= 0:
            raise BootstrapLockError("A bootstrap writer lock PID must be positive.")
        if not isinstance(self.hostname, str) or not self.hostname:
            raise BootstrapLockError("A bootstrap writer lock hostname is required.")

    def __enter__(self) -> BootstrapWriterLock:
        return self.acquire()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.release()

    def acquire(self) -> BootstrapWriterLock:
        """Acquire the lock or fail with explicit recovery guidance."""

        if self._owned_metadata is not None or self._owned_identity is not None:
            raise BootstrapLockError(f"Bootstrap writer lock {self.path} is already held.")

        metadata = self._new_metadata()
        serialized = (
            json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        recovered_stale = False
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            recovered_stale = self._handle_existing_lock()
            try:
                descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError:
                detail = (
                    " after stale recovery"
                    if recovered_stale
                    else " while the existing lock changed"
                )
                raise BootstrapLockError(
                    f"Another contender acquired bootstrap writer lock {self.path}"
                    f"{detail}; its lock was preserved. Retry in a new acquisition "
                    "attempt after verifying that writer's status."
                ) from None
            except OSError as error:
                raise BootstrapLockError(
                    f"Cannot create bootstrap writer lock {self.path}: {error}."
                ) from error
        except OSError as error:
            raise BootstrapLockError(
                f"Cannot create bootstrap writer lock {self.path}: {error}."
            ) from error

        identity: tuple[int, int] | None = None
        operation = "identity"
        failure: OSError | None = None
        try:
            identity = _stat_identity(os.fstat(descriptor))
            operation = "write"
            _write_all(descriptor, serialized, "lock")
            operation = "fsync"
            os.fsync(descriptor)
        except OSError as error:
            failure = error
        try:
            os.close(descriptor)
        except OSError as error:
            if failure is None:
                operation = "close"
                failure = error
            else:
                operation = f"{operation}/close"
                failure = OSError(f"{failure}; close also failed: {error}")

        if failure is not None:
            self._clear_ownership()
            if identity is None:
                raise BootstrapLockError(
                    f"Cannot establish the identity of created bootstrap writer "
                    f"lock {self.path}: {failure}; the created lock is preserved "
                    "and requires inspection because this process cannot prove it "
                    "still owns the path."
                ) from failure
            try:
                self._discard_created_lock(identity)
            except BootstrapLockError as cleanup_error:
                raise BootstrapLockError(
                    f"Cannot {operation} bootstrap writer lock {self.path}: "
                    f"{failure}; safe cleanup also failed: {cleanup_error}"
                ) from failure
            raise BootstrapLockError(
                f"Cannot {operation} bootstrap writer lock {self.path}: {failure}."
            ) from failure

        self._owned_metadata = metadata
        self._owned_identity = identity
        return self

    def release(self) -> bool:
        """Release only the exact lock inode and metadata acquired by this object."""

        owned_metadata = self._owned_metadata
        owned_identity = self._owned_identity
        if owned_metadata is None or owned_identity is None:
            self._clear_ownership()
            return False

        quarantine = self._quarantine_current("release")
        if quarantine is None:
            self._clear_ownership()
            return False
        try:
            current = self._read_lock_at(quarantine)
        except BootstrapLockError:
            self._clear_ownership()
            self._restore_quarantine(quarantine, "unverifiable release candidate")
            return False

        metadata, identity = current
        if metadata != owned_metadata or identity != owned_identity:
            self._clear_ownership()
            self._restore_quarantine(quarantine, "changed release candidate")
            return False

        try:
            quarantine.unlink()
        except OSError as error:
            self._clear_ownership()
            try:
                self._restore_quarantine(quarantine, "failed release cleanup")
            except BootstrapLockError as restore_error:
                raise BootstrapLockError(
                    f"Cannot release bootstrap writer lock {self.path}: {error}; "
                    f"safe restoration also failed: {restore_error}"
                ) from error
            raise BootstrapLockError(
                f"Cannot release bootstrap writer lock {self.path}: {error}; "
                "the lock was restored without deletion."
            ) from error
        self._clear_ownership()
        return True

    def _new_metadata(self) -> dict[str, Any]:
        started_at = self._clock()
        if not isinstance(started_at, datetime) or started_at.tzinfo is None:
            raise BootstrapLockError(
                "The bootstrap writer lock clock must return a timezone-aware datetime."
            )
        nonce = self._nonce_factory()
        if not isinstance(nonce, str) or not nonce:
            raise BootstrapLockError("The bootstrap writer lock nonce must be non-empty.")
        return {
            "version": LOCK_FORMAT_VERSION,
            "pid": self.pid,
            "hostname": self.hostname,
            "started_at": started_at.astimezone(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "nonce": nonce,
        }

    def _handle_existing_lock(self) -> bool:
        current = self._read_existing_lock()
        if current is None:
            return False
        metadata, expected_identity = current
        self._refuse_foreign_or_active(metadata)
        if not self.recover_stale:
            raise BootstrapLockError(
                f"Bootstrap writer lock {self.path} appears stale on this host "
                f"(absent PID {metadata['pid']}), but explicit stale recovery is "
                "required; retry with recover_stale=True after verifying the owner exited."
            )

        quarantine = self._quarantine_current("stale recovery")
        if quarantine is None:
            return False
        try:
            quarantined_metadata, quarantined_identity = self._read_lock_at(quarantine)
        except BootstrapLockError as error:
            self._restore_quarantine(quarantine, "unverifiable stale candidate")
            raise BootstrapLockError(
                f"Bootstrap writer lock {self.path} changed or became invalid during "
                "stale recovery; it was restored without deletion. Inspect it before "
                "retrying."
            ) from error

        if (
            quarantined_identity != expected_identity
            or quarantined_metadata != metadata
        ):
            changed_owner_error: BootstrapLockError | None = None
            try:
                self._refuse_foreign_or_active(quarantined_metadata)
            except BootstrapLockError as error:
                changed_owner_error = error
            self._restore_quarantine(quarantine, "changed stale candidate")
            if changed_owner_error is not None:
                raise changed_owner_error
            raise BootstrapLockError(
                f"Bootstrap writer lock {self.path} changed during stale recovery; "
                "the replacement was restored without deletion. Retry as a new "
                "acquisition attempt after verifying its owner."
            )

        try:
            self._refuse_foreign_or_active(quarantined_metadata)
        except BootstrapLockError:
            self._restore_quarantine(quarantine, "active or foreign stale candidate")
            raise

        try:
            quarantine.unlink()
        except OSError as error:
            try:
                self._restore_quarantine(quarantine, "failed stale cleanup")
            except BootstrapLockError as restore_error:
                raise BootstrapLockError(
                    f"Cannot recover stale bootstrap writer lock {self.path}: {error}; "
                    f"safe restoration also failed: {restore_error}"
                ) from error
            raise BootstrapLockError(
                f"Cannot recover stale bootstrap writer lock {self.path}: {error}; "
                "the stale lock was restored without deletion."
            ) from error
        return True

    def _refuse_foreign_or_active(self, metadata: dict[str, Any]) -> None:
        if metadata["hostname"] != self.hostname:
            raise BootstrapLockError(
                f"Bootstrap writer lock {self.path} belongs to a different host "
                f"({metadata['hostname']!r}); it cannot be recovered locally."
            )
        if self._pid_is_alive(metadata["pid"]):
            raise BootstrapLockError(
                f"Bootstrap writer lock {self.path} is active on host "
                f"{self.hostname!r} (PID {metadata['pid']}); wait for that writer to exit."
            )

    def _read_existing_lock(
        self,
    ) -> tuple[dict[str, Any], tuple[int, int]] | None:
        return self._read_lock_at(self.path, missing_ok=True)

    def _read_lock_at(
        self,
        path: Path,
        *,
        missing_ok: bool = False,
    ) -> tuple[dict[str, Any], tuple[int, int]] | None:
        try:
            with path.open("rb") as stream:
                raw = stream.read()
                identity = _stat_identity(os.fstat(stream.fileno()))
        except FileNotFoundError:
            if missing_ok:
                return None
            raise BootstrapLockError(
                f"Bootstrap writer lock candidate {path} disappeared while it was "
                "being verified; inspect sibling quarantine files before retrying."
            ) from None
        except OSError as error:
            raise BootstrapLockError(
                f"Cannot read bootstrap writer lock {path}: {error}."
            ) from error
        try:
            payload = json.loads(raw.decode("utf-8"))
            metadata = _validate_lock_metadata(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise BootstrapLockError(
                f"Bootstrap writer lock {path} is invalid or truncated; it "
                "will not be removed automatically. Inspect ownership and remove it "
                "manually only when safe."
            ) from error
        return metadata, identity

    def _quarantine_current(self, purpose: str) -> Path | None:
        quarantine = self._reserve_quarantine()
        try:
            os.replace(self.path, quarantine)
        except FileNotFoundError:
            try:
                quarantine.unlink()
            except OSError as cleanup_error:
                raise BootstrapLockError(
                    f"Bootstrap writer lock {self.path} disappeared during {purpose}, "
                    f"and reserved quarantine {quarantine} could not be cleaned: "
                    f"{cleanup_error}."
                ) from cleanup_error
            return None
        except OSError as error:
            try:
                quarantine.unlink()
            except OSError as cleanup_error:
                raise BootstrapLockError(
                    f"Cannot quarantine bootstrap writer lock {self.path} for "
                    f"{purpose}: {error}; reserved quarantine {quarantine} could "
                    f"not be cleaned: {cleanup_error}."
                ) from error
            raise BootstrapLockError(
                f"Cannot quarantine bootstrap writer lock {self.path} for "
                f"{purpose}: {error}."
            ) from error
        return quarantine

    def _reserve_quarantine(self) -> Path:
        for _attempt in range(16):
            quarantine = self.path.with_name(
                f".{self.path.name}.{uuid.uuid4().hex}.quarantine"
            )
            try:
                descriptor = os.open(
                    quarantine,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                continue
            except OSError as error:
                raise BootstrapLockError(
                    f"Cannot reserve a sibling quarantine for bootstrap writer lock "
                    f"{self.path}: {error}."
                ) from error
            try:
                os.close(descriptor)
            except OSError as error:
                raise BootstrapLockError(
                    f"Cannot close reserved sibling quarantine {quarantine}: "
                    f"{error}; inspect and remove it only when safe."
                ) from error
            return quarantine
        raise BootstrapLockError(
            f"Cannot reserve a unique sibling quarantine for bootstrap writer lock "
            f"{self.path}; inspect existing quarantine files before retrying."
        )

    def _restore_quarantine(self, quarantine: Path, reason: str) -> None:
        try:
            os.link(quarantine, self.path)
        except OSError as error:
            raise BootstrapLockError(
                f"Cannot safely restore bootstrap writer lock {self.path} after "
                f"{reason}: {error}; quarantine {quarantine} is preserved. Inspect "
                "both paths and restore the quarantined lock only when safe."
            ) from error
        try:
            quarantine.unlink()
        except OSError as error:
            raise BootstrapLockError(
                f"Bootstrap writer lock {self.path} was restored after {reason}, but "
                f"quarantine {quarantine} could not be removed: {error}; it is "
                "preserved as an additional link and requires manual cleanup."
            ) from error

    def _discard_created_lock(self, expected_identity: tuple[int, int]) -> None:
        quarantine = self._quarantine_current("failed lock creation cleanup")
        if quarantine is None:
            return
        try:
            with quarantine.open("rb") as stream:
                quarantined_identity = _stat_identity(os.fstat(stream.fileno()))
        except OSError as error:
            self._restore_quarantine(quarantine, "unverifiable failed lock creation")
            raise BootstrapLockError(
                f"Cannot verify failed bootstrap writer lock creation: {error}; "
                "the candidate was restored without deletion."
            ) from error
        if quarantined_identity != expected_identity:
            self._restore_quarantine(quarantine, "changed failed lock creation")
            return
        try:
            quarantine.unlink()
        except OSError as error:
            self._restore_quarantine(quarantine, "failed lock creation cleanup")
            raise BootstrapLockError(
                f"Cannot clean failed bootstrap writer lock creation: {error}; "
                "the candidate was restored without deletion."
            ) from error

    def _clear_ownership(self) -> None:
        self._owned_metadata = None
        self._owned_identity = None


def _validate_lock_metadata(payload: Any) -> dict[str, Any]:
    fields = {"version", "pid", "hostname", "started_at", "nonce"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("invalid lock metadata shape")
    version = payload["version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != LOCK_FORMAT_VERSION
    ):
        raise ValueError("unsupported lock metadata version")
    pid = payload["pid"]
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("invalid lock PID")
    hostname = payload["hostname"]
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("invalid lock hostname")
    started_at = payload["started_at"]
    if not isinstance(started_at, str):
        raise ValueError("invalid lock start time")
    parsed_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    if parsed_time.tzinfo is None:
        raise ValueError("lock start time must include a timezone")
    nonce = payload["nonce"]
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("invalid lock nonce")
    return {
        "version": version,
        "pid": pid,
        "hostname": hostname,
        "started_at": started_at,
        "nonce": nonce,
    }


def _stat_identity(file_stat: os.stat_result) -> tuple[int, int]:
    return file_stat.st_dev, file_stat.st_ino


def _pid_is_alive(pid: int) -> bool:
    """Probe PID liveness without using ``os.kill`` on Windows."""

    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        error_invalid_parameter = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        open_process.restype = ctypes.c_void_p
        get_exit_code_process = kernel32.GetExitCodeProcess
        get_exit_code_process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        get_exit_code_process.restype = ctypes.c_int
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
        handle = open_process(process_query_limited_information, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == error_invalid_parameter:
                return False
            # Access denied and unexpected probe failures fail closed as active.
            return True
        try:
            exit_code = ctypes.c_ulong()
            if not get_exit_code_process(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            close_handle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        return error.errno != errno.ESRCH
    return True


__all__ = [
    "LOCK_FORMAT_VERSION",
    "STATE_FORMAT_VERSION",
    "STAGE_ORDER",
    "BootstrapLockError",
    "BootstrapStateError",
    "BootstrapStateStore",
    "BootstrapWriterLock",
]
