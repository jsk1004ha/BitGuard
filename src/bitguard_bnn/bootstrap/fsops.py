"""Ownership-safe retirement of private bootstrap files.

Retirement deliberately keeps its private quarantine directory.  Under a
same-account namespace threat, deleting that directory later would introduce
the same name-substitution race this module is designed to avoid.  Operators
can inspect the path included in :class:`RetirementError` and recover or remove
it out of band.
"""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


_CREATE_ATTEMPTS = 16
_QUARANTINE_PREFIX = ".bitguard-retired-"
_ARTIFACT_NAME = "artifact"


class RetirementError(RuntimeError):
    """A path could not be retired without risking another writer's file."""

    def __init__(
        self,
        message: str,
        *,
        recovery_path: Path | None = None,
        restored: bool = False,
    ) -> None:
        super().__init__(message)
        self.recovery_path = recovery_path
        self.restored = restored


@dataclass(frozen=True, slots=True)
class RetirementArtifact:
    quarantine: Path
    artifact: Path
    storage_reclaimed: bool


def _validate_identity(expected: object) -> tuple[int, ...]:
    if not isinstance(expected, tuple) or len(expected) not in {2, 3}:
        raise TypeError("expected identity must be a two- or three-integer tuple.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in expected):
        raise TypeError("expected identity must contain integers, not booleans.")
    if any(value < 0 for value in expected):
        raise ValueError("expected identity values must be non-negative.")
    return expected


def _identity(value: os.stat_result, width: int) -> tuple[int, ...]:
    identity = (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode))
    return identity[:width]


def _lstat(path: Path, *, subject: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise RetirementError(
            f"Cannot inspect {subject} {path}: {error}.",
            recovery_path=path,
        ) from error


def _require_owned_regular(
    path: Path,
    expected: tuple[int, ...],
    *,
    subject: str,
) -> os.stat_result:
    value = _lstat(path, subject=subject)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise RetirementError(
            f"{subject.capitalize()} {path} is not a regular non-symlink file and was "
            "preserved.",
            recovery_path=path,
        )
    if _identity(value, len(expected)) != expected:
        raise RetirementError(
            f"{subject.capitalize()} {path} has a foreign identity and was preserved.",
            recovery_path=path,
        )
    return value


def _reserve_quarantine(parent: Path) -> tuple[Path, tuple[int, int, int]]:
    for _attempt in range(_CREATE_ATTEMPTS):
        quarantine = parent / f"{_QUARANTINE_PREFIX}{uuid.uuid4().hex}"
        try:
            os.mkdir(quarantine, 0o700)
        except FileExistsError:
            continue
        except OSError as error:
            raise RetirementError(
                f"Cannot create a private retirement quarantine in {parent}: {error}.",
                recovery_path=parent,
            ) from error
        try:
            value = quarantine.lstat()
        except OSError as error:
            raise RetirementError(
                f"Cannot pin newly reserved retirement quarantine {quarantine}: {error}; "
                "the path was preserved for recovery.",
                recovery_path=quarantine,
            ) from error
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise RetirementError(
                f"New retirement quarantine {quarantine} changed type and was preserved "
                "for recovery.",
                recovery_path=quarantine,
            )
        if os.name != "nt" and stat.S_IMODE(value.st_mode) & 0o077:
            raise RetirementError(
                f"New retirement quarantine {quarantine} is not private mode 0700 and "
                "was preserved for recovery.",
                recovery_path=quarantine,
            )
        return quarantine, _directory_identity(value)
    raise RetirementError(
        f"Cannot reserve a unique private retirement quarantine in {parent}.",
        recovery_path=parent,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode))


def _pin_quarantine(
    quarantine: Path,
    reserved_identity: tuple[int, int, int],
) -> int | None:
    value = _lstat(quarantine, subject="retirement quarantine")
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise RetirementError(
            f"Retirement quarantine {quarantine} is not a private directory; it was "
            "preserved for recovery.",
            recovery_path=quarantine,
        )
    if _directory_identity(value) != reserved_identity:
        raise RetirementError(
            f"Retirement quarantine {quarantine} changed after reservation and was "
            "preserved for recovery.",
            recovery_path=quarantine,
        )
    if os.name == "nt":
        return None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(quarantine, flags)
    except OSError as error:
        raise RetirementError(
            f"Cannot pin retirement quarantine {quarantine}: {error}; it was preserved "
            "for recovery.",
            recovery_path=quarantine,
        ) from error
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise RetirementError(
            f"Cannot validate opened retirement quarantine {quarantine}: {error}; it was "
            "preserved for recovery.",
            recovery_path=quarantine,
        ) from error
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    if _directory_identity(opened) != reserved_identity:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise RetirementError(
            f"Retirement quarantine {quarantine} changed while it was opened and was "
            "preserved for recovery.",
            recovery_path=quarantine,
        )
    return descriptor


def _validate_quarantine(
    quarantine: Path,
    expected: tuple[int, int, int],
    descriptor: int | None,
) -> None:
    value = _lstat(quarantine, subject="retirement quarantine")
    if _directory_identity(value) != expected or not stat.S_ISDIR(value.st_mode):
        raise RetirementError(
            f"Retirement quarantine {quarantine} changed identity; all paths were "
            "preserved for recovery.",
            recovery_path=quarantine,
        )
    if descriptor is not None:
        try:
            pinned = os.fstat(descriptor)
        except OSError as error:
            raise RetirementError(
                f"Cannot validate pinned retirement quarantine {quarantine}: {error}; "
                "all paths were preserved for recovery.",
                recovery_path=quarantine,
            ) from error
        if _directory_identity(pinned) != expected:
            raise RetirementError(
                f"Pinned retirement quarantine {quarantine} changed identity; all paths "
                "were preserved for recovery.",
                recovery_path=quarantine,
            )


def _restore_foreign_artifact(artifact: Path, original: Path) -> bool:
    try:
        original.lstat()
    except FileNotFoundError:
        pass
    except OSError:
        return False
    else:
        return False
    try:
        os.link(artifact, original, follow_symlinks=False)
    except (FileExistsError, NotImplementedError, OSError):
        return False
    try:
        artifact_stat = artifact.lstat()
        restored_stat = original.lstat()
    except OSError:
        return False
    return _directory_identity(artifact_stat) == _directory_identity(restored_stat)


def _rename_noreplace(
    source: Path,
    destination: Path,
    destination_directory_descriptor: int | None,
) -> None:
    """Rename without replacing a concurrently inserted destination.

    Python exposes the required no-clobber behavior on Windows.  Linux uses
    ``renameat2(RENAME_NOREPLACE)`` and macOS uses
    ``renameatx_np(RENAME_EXCL)``.  Other POSIX kernels fail closed instead of
    falling back to the clobbering semantics of ``rename(2)``.
    """

    if os.name == "nt":
        os.rename(source, destination)
        return
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renameatx = getattr(libc, "renameatx_np", None)
        if renameatx is None:
            raise OSError(errno.ENOTSUP, "renameatx_np is unavailable")
        renameatx.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx.restype = ctypes.c_int
        at_fdcwd = -2
        destination_directory = (
            destination_directory_descriptor
            if destination_directory_descriptor is not None
            else at_fdcwd
        )
        destination_name = (
            os.fsencode(destination.name)
            if destination_directory_descriptor is not None
            else os.fsencode(destination)
        )
        result = renameatx(
            at_fdcwd,
            os.fsencode(source),
            destination_directory,
            destination_name,
            0x00000004,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        raise OSError(error_number, os.strerror(error_number), source)
    if not sys.platform.startswith("linux"):
        raise OSError(
            errno.ENOTSUP,
            "atomic no-clobber rename is unavailable on this platform",
        )

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "renameat2 is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    destination_directory = (
        destination_directory_descriptor
        if destination_directory_descriptor is not None
        else at_fdcwd
    )
    destination_name = (
        os.fsencode(destination.name)
        if destination_directory_descriptor is not None
        else os.fsencode(destination)
    )
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        destination_directory,
        destination_name,
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), source)


def _reclaim_retired_storage(
    artifact: Path,
    expected: tuple[int, ...],
    quarantine: Path,
    quarantine_identity: tuple[int, int, int],
    quarantine_descriptor: int | None,
) -> None:
    _validate_quarantine(quarantine, quarantine_identity, quarantine_descriptor)
    before = _require_owned_regular(
        artifact,
        expected,
        subject="retired storage artifact",
    )
    if before.st_nlink != 1:
        raise RetirementError(
            f"Retired storage artifact {artifact} has {before.st_nlink} hard links; "
            "storage was preserved because truncation could affect another path.",
            recovery_path=artifact,
        )
    flags = (
        os.O_WRONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    failure: BaseException | None = None
    try:
        descriptor = os.open(artifact, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened, len(expected)) != expected:
            raise RetirementError(
                f"Retired storage artifact {artifact} changed before descriptor validation; "
                "the foreign recovery path was preserved.",
                recovery_path=artifact,
            )
        if opened.st_nlink != 1:
            raise RetirementError(
                f"Retired storage artifact {artifact} gained another hard link; storage "
                "was preserved.",
                recovery_path=artifact,
            )
        _require_owned_regular(
            artifact,
            expected,
            subject="retired storage artifact",
        )
        _validate_quarantine(quarantine, quarantine_identity, quarantine_descriptor)
        os.ftruncate(descriptor, 0)
        os.fsync(descriptor)
        _require_owned_regular(
            artifact,
            expected,
            subject="reclaimed retirement artifact",
        )
    except OSError as error:
        failure = error
        raise RetirementError(
            f"Cannot reclaim proven retired storage at {artifact}: {error}; the recovery "
            "artifact was preserved.",
            recovery_path=artifact,
        ) from error
    except BaseException as error:
        failure = error
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as close_error:
                if failure is None:
                    raise RetirementError(
                        f"Retired artifact {artifact} was reclaimed but its descriptor "
                        f"could not be closed: {close_error}.",
                        recovery_path=artifact,
                    ) from close_error


def retire_owned_path(
    path: Path | str,
    expected_identity: tuple[int, ...],
    *,
    purpose: str,
    reclaim_storage: bool = False,
) -> RetirementArtifact | None:
    """Move a proven writer-owned path into a retained private quarantine.

    The original candidate is never unlinked.  A replacement moved by the
    final rename is retained in quarantine and restored with a no-clobber hard
    link when the original name is absent.
    """

    expected = _validate_identity(expected_identity)
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("retirement purpose must be a non-empty string.")
    if not isinstance(reclaim_storage, bool):
        raise TypeError("reclaim_storage must be a boolean.")
    raw_candidate = Path(path)
    try:
        candidate = raw_candidate.parent.resolve(strict=True) / raw_candidate.name
    except OSError as error:
        raise RetirementError(
            f"Cannot resolve the retirement parent for {raw_candidate}: {error}.",
            recovery_path=raw_candidate,
        ) from error
    try:
        initial = candidate.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RetirementError(
            f"Cannot inspect {purpose} candidate {candidate}: {error}.",
            recovery_path=candidate,
        ) from error
    if (
        stat.S_ISLNK(initial.st_mode)
        or not stat.S_ISREG(initial.st_mode)
        or _identity(initial, len(expected)) != expected
    ):
        raise RetirementError(
            f"The {purpose} candidate {candidate} has a foreign identity and was preserved.",
            recovery_path=candidate,
        )

    quarantine, quarantine_identity = _reserve_quarantine(candidate.parent)
    quarantine_descriptor = _pin_quarantine(quarantine, quarantine_identity)
    artifact = quarantine / _ARTIFACT_NAME
    active_error: BaseException | None = None
    try:
        _validate_quarantine(quarantine, quarantine_identity, quarantine_descriptor)
        try:
            artifact.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise RetirementError(
                f"Cannot inspect retirement destination {artifact}: {error}; the source "
                "and quarantine were preserved for recovery.",
                recovery_path=quarantine,
            ) from error
        else:
            raise RetirementError(
                f"Retirement destination collision at {artifact}; the source and existing "
                "recovery artifact were preserved.",
                recovery_path=artifact,
            )
        _require_owned_regular(candidate, expected, subject=f"{purpose} candidate")
        try:
            _rename_noreplace(candidate, artifact, quarantine_descriptor)
        except FileExistsError as error:
            raise RetirementError(
                f"Retirement destination collision at {artifact}; inspect the retained "
                "recovery path before retrying.",
                recovery_path=artifact,
            ) from error
        except OSError as error:
            raise RetirementError(
                f"Cannot atomically retire {candidate} into {artifact}: {error}; inspect "
                "both recovery paths before retrying.",
                recovery_path=artifact if artifact.exists() else quarantine,
            ) from error

        _validate_quarantine(quarantine, quarantine_identity, quarantine_descriptor)
        moved = _lstat(artifact, subject="retired artifact")
        if _identity(moved, len(expected)) != expected or not stat.S_ISREG(moved.st_mode):
            restored = _restore_foreign_artifact(artifact, candidate)
            status = "restored without overwrite" if restored else "not restored"
            raise RetirementError(
                f"A foreign replacement was moved while retiring {candidate}; it was "
                f"preserved at recovery path {artifact} and {status} at the original path.",
                recovery_path=artifact,
                restored=restored,
            )

        if reclaim_storage:
            _reclaim_retired_storage(
                artifact,
                expected,
                quarantine,
                quarantine_identity,
                quarantine_descriptor,
            )
        return RetirementArtifact(
            quarantine=quarantine,
            artifact=artifact,
            storage_reclaimed=reclaim_storage,
        )
    except BaseException as error:
        active_error = error
        raise
    finally:
        if quarantine_descriptor is not None:
            try:
                os.close(quarantine_descriptor)
            except OSError as close_error:
                if active_error is None:
                    raise RetirementError(
                        f"Retirement quarantine {quarantine} could not be closed: "
                        f"{close_error}; it was retained for recovery.",
                        recovery_path=quarantine,
                    ) from close_error
