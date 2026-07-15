from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import bitguard_bnn.bootstrap.state as bootstrap_state
from bitguard_bnn.bootstrap.state import (
    LOCK_FORMAT_VERSION,
    STATE_FORMAT_VERSION,
    STAGE_ORDER,
    BootstrapLockError,
    BootstrapStateError,
    BootstrapStateStore,
    BootstrapWriterLock,
)


FIXED_TIME = datetime(2026, 7, 14, 4, 30, tzinfo=timezone.utc)


class BootstrapStateStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "bootstrap"
        self.root.mkdir()
        self.state_path = self.root / "state.json"

    def output(self, name: str, content: bytes = b"complete") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def state_temporary_paths(self) -> list[Path]:
        return sorted(self.root.glob(f"{self.state_path.name}.*.tmp"))

    def test_absent_state_starts_empty(self):
        store = BootstrapStateStore(self.state_path)

        self.assertEqual(store.completed_stages, ())

    def test_stage_is_reused_only_when_input_signature_and_outputs_match(self):
        root = self.root
        store = BootstrapStateStore(root / "state.json")
        output = root / "archive.zip"
        output.write_bytes(b"complete")
        store.complete("acquire", "input-a", [output])
        self.assertTrue(store.reusable("acquire", "input-a"))
        output.write_bytes(b"changed")
        self.assertFalse(store.reusable("acquire", "input-a"))

    def test_reuse_detects_same_size_content_change(self):
        output = self.output("archive.zip", b"first")
        store = BootstrapStateStore(self.state_path)
        store.complete("acquire", "input-a", [output])

        output.write_bytes(b"other")

        self.assertFalse(store.reusable("acquire", "input-a"))

    def test_reuse_detects_size_change_missing_output_and_signature_mismatch(self):
        output = self.output("archive.zip", b"first")
        store = BootstrapStateStore(self.state_path)
        store.complete("acquire", "input-a", [output])

        self.assertFalse(store.reusable("acquire", "input-b"))
        output.write_bytes(b"longer")
        self.assertFalse(store.reusable("acquire", "input-a"))
        output.unlink()
        self.assertFalse(store.reusable("acquire", "input-a"))

    def test_fingerprint_is_relative_portable_and_uses_sha256(self):
        output = self.output("downloads/archive.zip", b"payload")
        store = BootstrapStateStore(self.state_path)

        store.complete("acquire", "input-a", [output])

        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        fingerprint = payload["stages"]["acquire"]["outputs"][0]
        self.assertEqual(fingerprint["path"], "downloads/archive.zip")
        self.assertNotIn("\\", fingerprint["path"])
        self.assertEqual(fingerprint["size"], len(b"payload"))
        self.assertEqual(
            fingerprint["sha256"], hashlib.sha256(b"payload").hexdigest()
        )

    def test_complete_requires_existing_regular_files(self):
        store = BootstrapStateStore(self.state_path)
        directory = self.root / "directory"
        directory.mkdir()

        for output in (self.root / "missing.zip", directory):
            with self.subTest(output=output):
                with self.assertRaisesRegex(BootstrapStateError, "regular file"):
                    store.complete("acquire", "input-a", [output])

        self.assertEqual(store.completed_stages, ())

    def test_complete_rejects_output_outside_state_root(self):
        outside = Path(self.temporary.name) / "outside.zip"
        outside.write_bytes(b"payload")
        store = BootstrapStateStore(self.state_path)

        with self.assertRaisesRegex(BootstrapStateError, "state root"):
            store.complete("acquire", "input-a", [outside])

        self.assertFalse(self.state_path.exists())

    def test_unknown_future_format_version_is_rejected(self):
        self.state_path.write_text(
            json.dumps({"version": STATE_FORMAT_VERSION + 1, "stages": {}}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(BootstrapStateError, "newer.*upgrade"):
            BootstrapStateStore(self.state_path)

    def test_malformed_state_is_rejected_with_actionable_error(self):
        malformed_payloads = (
            "{truncated",
            json.dumps({"version": STATE_FORMAT_VERSION}),
            json.dumps(
                {
                    "version": STATE_FORMAT_VERSION,
                    "stages": {"unknown": {}},
                }
            ),
            json.dumps(
                {
                    "version": STATE_FORMAT_VERSION,
                    "stages": {
                        "acquire": {
                            "input_signature": "input-a",
                            "outputs": [
                                {"path": "../escape", "size": 1, "sha256": "0" * 64}
                            ],
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "version": STATE_FORMAT_VERSION,
                    "stages": {
                        "acquire": {
                            "input_signature": "input-a",
                            "outputs": [
                                {
                                    "path": "C:/machine-specific/archive.zip",
                                    "size": 1,
                                    "sha256": "0" * 64,
                                }
                            ],
                        }
                    },
                }
            ),
        )

        for payload in malformed_payloads:
            with self.subTest(payload=payload):
                self.state_path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(BootstrapStateError, "state"):
                    BootstrapStateStore(self.state_path)

    def test_state_rejects_control_characters_in_stored_output_paths(self):
        for unsafe_path in ("archive\x00.zip", "archive\n.zip", "archive\x7f.zip"):
            with self.subTest(path=unsafe_path):
                self.state_path.write_text(
                    json.dumps(
                        {
                            "version": STATE_FORMAT_VERSION,
                            "stages": {
                                "acquire": {
                                    "input_signature": "input-a",
                                    "outputs": [
                                        {
                                            "path": unsafe_path,
                                            "size": 1,
                                            "sha256": "0" * 64,
                                        }
                                    ],
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(BootstrapStateError, "unsafe.*path"):
                    BootstrapStateStore(self.state_path)

    def test_path_resolution_value_error_is_wrapped_as_state_error(self):
        store = BootstrapStateStore(self.state_path)
        output = self.output("archive.zip")

        with patch.object(
            Path,
            "resolve",
            autospec=True,
            side_effect=ValueError("embedded null character"),
        ):
            with self.assertRaisesRegex(BootstrapStateError, "regular file"):
                store.complete("acquire", "input-a", [output])

    def test_persistence_fsyncs_file_then_replaces_then_fsyncs_parent(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        events: list[str] = []
        real_fsync = os.fsync
        real_replace = Path.replace

        with (
            patch(
                "bitguard_bnn.bootstrap.state.Path.replace",
                autospec=True,
                side_effect=lambda source, target: (
                    events.append("replace"),
                    real_replace(source, target),
                )[1],
            ),
            patch(
                "bitguard_bnn.bootstrap.state.os.fsync",
                side_effect=lambda descriptor: (
                    events.append("file-fsync"),
                    real_fsync(descriptor),
                )[1],
            ),
            patch(
                "bitguard_bnn.bootstrap.state._fsync_parent_directory",
                side_effect=lambda _path: events.append("directory-fsync"),
            ),
        ):
            store.complete("acquire", "input-a", [output])

        self.assertEqual(events, ["file-fsync", "replace", "directory-fsync"])
        self.assertEqual(self.state_temporary_paths(), [])

    def test_late_substitution_of_predictable_temp_cannot_corrupt_state_commit(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        predictable = self.root / "state.json.tmp"
        hostile = self.root / "hostile.tmp"
        hostile_bytes = b"{foreign state"
        hostile.write_bytes(hostile_bytes)
        real_close = os.close
        substituted = False

        def substitute_after_close(descriptor: int) -> None:
            nonlocal substituted
            real_close(descriptor)
            if not substituted:
                substituted = True
                os.replace(hostile, predictable)

        with patch(
            "bitguard_bnn.bootstrap.state.os.close",
            side_effect=substitute_after_close,
        ):
            store.complete("acquire", "input-a", [output])

        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages,
            ("acquire",),
        )
        self.assertEqual(predictable.read_bytes(), hostile_bytes)
        self.assertNotEqual(self.state_path.read_bytes(), hostile_bytes)

    def test_state_temp_uses_restrictive_exclusive_nofollow_flags(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        real_open = os.open
        calls: list[tuple[Path, int, int]] = []

        def recording_open(path, flags, mode=0o777):
            candidate = Path(path)
            if (
                candidate.parent == self.root
                and candidate.name.startswith(f"{self.state_path.name}.")
                and candidate.name.endswith(".tmp")
            ):
                calls.append((candidate, flags, mode))
            return real_open(path, flags, mode)

        with patch("bitguard_bnn.bootstrap.state.os.open", side_effect=recording_open):
            store.complete("acquire", "input-a", [output])

        self.assertEqual(len(calls), 1)
        temporary, flags, mode = calls[0]
        token = temporary.name.removeprefix(f"{self.state_path.name}.").removesuffix(
            ".tmp"
        )
        self.assertRegex(token, r"^[0-9a-f]{32}$")
        required_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        self.assertEqual(flags & required_flags, required_flags)
        self.assertEqual(mode, 0o600)
        if hasattr(os, "O_NOFOLLOW"):
            self.assertTrue(flags & os.O_NOFOLLOW)

    def test_preexisting_predictable_state_temp_is_ignored_and_preserved(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        temporary = self.root / "state.json.tmp"
        hostile_bytes = b"pre-existing contender"
        temporary.write_bytes(hostile_bytes)

        store.complete("acquire", "input-a", [output])

        self.assertEqual(temporary.read_bytes(), hostile_bytes)
        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages,
            ("acquire",),
        )

    def test_private_state_temp_name_collision_retries_without_clobber(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        collision_token = bootstrap_state.uuid.UUID(int=1)
        private_token = bootstrap_state.uuid.UUID(int=2)
        collision = self.root / f"state.json.{collision_token.hex}.tmp"
        hostile_bytes = b"pre-existing private-name collision"
        collision.write_bytes(hostile_bytes)

        with patch(
            "bitguard_bnn.bootstrap.state.uuid.uuid4",
            side_effect=(collision_token, private_token),
        ) as token_factory:
            store.complete("acquire", "input-a", [output])

        self.assertEqual(token_factory.call_count, 2)
        self.assertEqual(collision.read_bytes(), hostile_bytes)
        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages,
            ("acquire",),
        )
        self.assertEqual(self.state_temporary_paths(), [collision])

    def test_private_state_temp_collision_exhaustion_is_bounded_and_actionable(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        collision_token = bootstrap_state.uuid.UUID(int=3)
        collision = self.root / f"state.json.{collision_token.hex}.tmp"
        hostile_bytes = b"persistent collision"
        collision.write_bytes(hostile_bytes)

        with patch(
            "bitguard_bnn.bootstrap.state.uuid.uuid4",
            return_value=collision_token,
        ) as token_factory:
            with self.assertRaisesRegex(
                BootstrapStateError,
                r"unique private.*16.*collisions.*inspect",
            ):
                store.complete("acquire", "input-a", [output])

        self.assertEqual(token_factory.call_count, 16)
        self.assertEqual(collision.read_bytes(), hostile_bytes)
        self.assertFalse(self.state_path.exists())

    def test_private_state_temp_creation_does_not_retry_non_collision_errors(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        real_open = os.open
        attempts = 0

        def fail_private_open(path, flags, mode=0o777):
            nonlocal attempts
            candidate = Path(path)
            if candidate.name.startswith("state.json.") and candidate.name.endswith(
                ".tmp"
            ):
                attempts += 1
                raise PermissionError("simulated denied private temp creation")
            return real_open(path, flags, mode)

        with patch(
            "bitguard_bnn.bootstrap.state.os.open",
            side_effect=fail_private_open,
        ):
            with self.assertRaisesRegex(BootstrapStateError, "create.*denied"):
                store.complete("acquire", "input-a", [output])

        self.assertEqual(attempts, 1)
        self.assertFalse(self.state_path.exists())

    def test_restart_invalidates_stage_and_dependants(self):
        store = BootstrapStateStore(self.state_path)
        for stage in ("acquire", "extract", "inspect"):
            store.complete(stage, f"input-{stage}", [self.output(f"{stage}.out")])

        store.invalidate_from("extract", STAGE_ORDER)

        self.assertIn("acquire", store.completed_stages)
        self.assertNotIn("extract", store.completed_stages)
        self.assertNotIn("inspect", store.completed_stages)
        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages, ("acquire",)
        )

    def test_invalidate_rejects_unknown_stage_and_inconsistent_order(self):
        store = BootstrapStateStore(self.state_path)

        with self.assertRaisesRegex(BootstrapStateError, "unknown stage"):
            store.invalidate_from("download", STAGE_ORDER)
        with self.assertRaisesRegex(BootstrapStateError, "canonical stage order"):
            store.invalidate_from("extract", ("acquire", "extract", "inspect"))

    def test_preexisting_predictable_state_temp_symlink_is_ignored(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        temporary = self.root / "state.json.tmp"
        target = self.root / "hostile-target"
        target.write_bytes(b"do not modify")
        try:
            temporary.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation unavailable: {error}")

        store.complete("acquire", "input-a", [output])

        self.assertTrue(temporary.is_symlink())
        self.assertEqual(target.read_bytes(), b"do not modify")
        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages,
            ("acquire",),
        )

    def test_partial_state_writes_are_completed(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        real_write = os.write
        write_sizes: list[int] = []

        def partial_write(descriptor: int, data: bytes) -> int:
            chunk = data[:7]
            write_sizes.append(len(chunk))
            return real_write(descriptor, chunk)

        with patch("bitguard_bnn.bootstrap.state.os.write", side_effect=partial_write):
            store.complete("acquire", "input-a", [output])

        self.assertGreater(len(write_sizes), 1)
        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages,
            ("acquire",),
        )

    def test_zero_progress_state_write_preserves_previous_state_and_cleans_temp(self):
        self._assert_descriptor_failure_preserves_previous_state(
            "write",
            patch("bitguard_bnn.bootstrap.state.os.write", return_value=0),
        )

    def test_failed_atomic_replace_preserves_previous_state_and_cleans_temp(self):
        archive = self.output("archive.zip")
        inspection = self.output("inspection.json")
        store = BootstrapStateStore(self.state_path)
        store.complete("acquire", "input-a", [archive])
        previous_bytes = self.state_path.read_bytes()

        with patch(
            "bitguard_bnn.bootstrap.state.Path.replace",
            side_effect=OSError("simulated replace failure"),
        ):
            with self.assertRaisesRegex(BootstrapStateError, "persist"):
                store.complete("inspect", "input-b", [inspection])

        self.assertEqual(self.state_path.read_bytes(), previous_bytes)
        self.assertEqual(store.completed_stages, ("acquire",))
        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages, ("acquire",)
        )
        self.assertEqual(self.state_temporary_paths(), [])

    def test_post_replace_verification_rejects_foreign_inode(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        foreign = self.root / "foreign-state.json"
        foreign_bytes = b"foreign replacement"
        foreign.write_bytes(foreign_bytes)
        real_replace = Path.replace

        def replace_then_substitute(source: Path, target: Path) -> Path:
            result = real_replace(source, target)
            os.replace(foreign, target)
            return result

        with patch(
            "bitguard_bnn.bootstrap.state.Path.replace",
            autospec=True,
            side_effect=replace_then_substitute,
        ):
            with self.assertRaisesRegex(BootstrapStateError, "verify.*identity"):
                store.complete("acquire", "input-a", [output])

        self.assertEqual(store.completed_stages, ())
        self.assertEqual(self.state_path.read_bytes(), foreign_bytes)

    def test_post_replace_verification_rejects_same_inode_content_change(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        foreign_bytes = b"same inode, foreign content"
        real_replace = Path.replace

        def replace_then_modify(source: Path, target: Path) -> Path:
            result = real_replace(source, target)
            Path(target).write_bytes(foreign_bytes)
            return result

        with patch(
            "bitguard_bnn.bootstrap.state.Path.replace",
            autospec=True,
            side_effect=replace_then_modify,
        ):
            with self.assertRaisesRegex(BootstrapStateError, "verify.*content"):
                store.complete("acquire", "input-a", [output])

        self.assertEqual(store.completed_stages, ())
        self.assertEqual(self.state_path.read_bytes(), foreign_bytes)

    def _assert_descriptor_failure_preserves_previous_state(
        self, operation: str, failure_patch
    ) -> None:
        archive = self.output("archive.zip")
        inspection = self.output("inspection.json")
        store = BootstrapStateStore(self.state_path)
        store.complete("acquire", "input-a", [archive])
        previous_bytes = self.state_path.read_bytes()
        with failure_patch:
            with self.assertRaisesRegex(BootstrapStateError, operation):
                store.complete("inspect", "input-b", [inspection])

        self.assertEqual(self.state_path.read_bytes(), previous_bytes)
        self.assertEqual(store.completed_stages, ("acquire",))
        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages, ("acquire",)
        )
        self.assertEqual(self.state_temporary_paths(), [])

    def test_failed_state_write_preserves_previous_state_and_cleans_temp(self):
        self._assert_descriptor_failure_preserves_previous_state(
            "write",
            patch(
                "bitguard_bnn.bootstrap.state.os.write",
                side_effect=OSError("simulated state write failure"),
            ),
        )

    def test_failed_state_fsync_preserves_previous_state_and_cleans_temp(self):
        self._assert_descriptor_failure_preserves_previous_state(
            "fsync",
            patch(
                "bitguard_bnn.bootstrap.state.os.fsync",
                side_effect=OSError("simulated state fsync failure"),
            ),
        )

    def test_failed_state_close_preserves_previous_state_and_cleans_temp(self):
        real_close = os.close
        failed = False

        def close_then_fail_once(descriptor: int) -> None:
            nonlocal failed
            real_close(descriptor)
            if not failed:
                failed = True
                raise OSError("simulated state close failure")

        self._assert_descriptor_failure_preserves_previous_state(
            "close",
            patch(
                "bitguard_bnn.bootstrap.state.os.close",
                side_effect=close_then_fail_once,
            ),
        )

    def test_failed_state_identity_check_fails_closed_and_preserves_temp(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)

        with patch(
            "bitguard_bnn.bootstrap.state.os.fstat",
            side_effect=OSError("simulated state fstat failure"),
        ):
            with self.assertRaisesRegex(BootstrapStateError, "identity.*inspect"):
                store.complete("acquire", "input-a", [output])

        temporary_paths = self.state_temporary_paths()
        self.assertEqual(len(temporary_paths), 1)
        self.assertTrue(temporary_paths[0].is_file())
        self.assertFalse(self.state_path.exists())

    def test_failed_cleanup_does_not_delete_replacement_temp(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)
        hostile = self.root / "hostile.tmp"
        hostile_bytes = b"late contender"
        hostile.write_bytes(hostile_bytes)
        real_open = os.open
        real_close = os.close
        created: list[Path] = []
        failed = False

        def record_private_open(path, flags, mode=0o777):
            candidate = Path(path)
            if candidate.name.startswith("state.json.") and candidate.name.endswith(
                ".tmp"
            ):
                created.append(candidate)
            return real_open(path, flags, mode)

        def close_replace_then_fail(descriptor: int) -> None:
            nonlocal failed
            real_close(descriptor)
            if not failed:
                failed = True
                os.replace(hostile, created[0])
                raise OSError("simulated close race")

        with (
            patch(
                "bitguard_bnn.bootstrap.state.os.open",
                side_effect=record_private_open,
            ),
            patch(
                "bitguard_bnn.bootstrap.state.os.close",
                side_effect=close_replace_then_fail,
            ),
        ):
            with self.assertRaisesRegex(BootstrapStateError, "close"):
                store.complete("acquire", "input-a", [output])

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].read_bytes(), hostile_bytes)
        self.assertFalse(self.state_path.exists())


class ParentDirectoryFsyncTest(unittest.TestCase):
    def test_posix_parent_directory_is_opened_fsynced_and_closed(self):
        parent = Path("parent")
        expected_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        with (
            patch.object(bootstrap_state, "_DIRECTORY_FSYNC_SUPPORTED", True),
            patch("bitguard_bnn.bootstrap.state.os.open", return_value=71) as open_,
            patch("bitguard_bnn.bootstrap.state.os.fsync") as fsync,
            patch("bitguard_bnn.bootstrap.state.os.close") as close,
        ):
            bootstrap_state._fsync_parent_directory(parent / "state.json")

        open_.assert_called_once_with(parent, expected_flags)
        fsync.assert_called_once_with(71)
        close.assert_called_once_with(71)

    def test_parent_directory_open_fsync_and_close_failures_are_not_swallowed(self):
        cases = ("open", "fsync", "close")
        for operation in cases:
            with self.subTest(operation=operation):
                with (
                    patch.object(bootstrap_state, "_DIRECTORY_FSYNC_SUPPORTED", True),
                    patch(
                        "bitguard_bnn.bootstrap.state.os.open",
                        return_value=72,
                        side_effect=OSError("open failed") if operation == "open" else None,
                    ),
                    patch(
                        "bitguard_bnn.bootstrap.state.os.fsync",
                        side_effect=OSError("fsync failed") if operation == "fsync" else None,
                    ),
                    patch(
                        "bitguard_bnn.bootstrap.state.os.close",
                        side_effect=OSError("close failed") if operation == "close" else None,
                    ) as close,
                ):
                    with self.assertRaisesRegex(OSError, operation):
                        bootstrap_state._fsync_parent_directory(Path("parent/state.json"))
                if operation == "fsync":
                    close.assert_called_once_with(72)

    def test_windows_directory_fsync_unavailability_is_explicit_noop(self):
        with (
            patch.object(bootstrap_state, "_DIRECTORY_FSYNC_SUPPORTED", False),
            patch("bitguard_bnn.bootstrap.state.os.open") as open_,
            patch("bitguard_bnn.bootstrap.state.os.fsync") as fsync,
        ):
            bootstrap_state._fsync_parent_directory(Path("parent/state.json"))

        open_.assert_not_called()
        fsync.assert_not_called()

class BootstrapWriterLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.lock_path = Path(self.temporary.name) / "bootstrap.lock"

    @staticmethod
    def metadata(
        *,
        pid: int = 101,
        hostname: str = "test-host",
        started_at: str = "2026-07-14T04:30:00Z",
        nonce: str = "existing-owner",
    ) -> dict[str, object]:
        return {
            "version": LOCK_FORMAT_VERSION,
            "pid": pid,
            "hostname": hostname,
            "started_at": started_at,
            "nonce": nonce,
        }

    def write_lock(self, **overrides: object) -> None:
        self.write_lock_at(self.lock_path, **overrides)

    def write_lock_at(self, path: Path, **overrides: object) -> None:
        metadata = self.metadata(**overrides)
        path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    def quarantine_paths(self) -> list[Path]:
        return list(
            self.lock_path.parent.glob(f".{self.lock_path.name}.*.quarantine")
        )

    def lock(self, **overrides: object) -> BootstrapWriterLock:
        arguments = {
            "pid": 202,
            "hostname": "test-host",
            "clock": lambda: FIXED_TIME,
            "nonce_factory": lambda: "new-owner",
            "pid_is_alive": lambda _pid: True,
        }
        arguments.update(overrides)
        return BootstrapWriterLock(self.lock_path, **arguments)

    def test_exclusive_acquisition_contention_and_release(self):
        first = self.lock(nonce_factory=lambda: "first-owner")
        first.acquire()
        second = self.lock(nonce_factory=lambda: "second-owner")

        with self.assertRaisesRegex(BootstrapLockError, "active.*PID 202"):
            second.acquire()

        self.assertTrue(self.lock_path.exists())
        self.assertTrue(first.release())
        self.assertFalse(self.lock_path.exists())

    def test_context_manager_records_injected_metadata_and_releases(self):
        with self.lock() as acquired:
            self.assertIsInstance(acquired, BootstrapWriterLock)
            metadata = json.loads(self.lock_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["pid"], 202)
            self.assertEqual(metadata["hostname"], "test-host")
            self.assertEqual(metadata["started_at"], "2026-07-14T04:30:00Z")
            self.assertEqual(metadata["nonce"], "new-owner")

        self.assertFalse(self.lock_path.exists())

    def test_fstat_failure_closes_descriptor_and_preserves_unverified_lock(self):
        lock = self.lock()
        real_close = os.close

        with (
            patch(
                "bitguard_bnn.bootstrap.state.os.fstat",
                side_effect=OSError("simulated lock fstat failure"),
            ),
            patch(
                "bitguard_bnn.bootstrap.state.os.close",
                side_effect=real_close,
            ) as close,
        ):
            with self.assertRaisesRegex(
                BootstrapLockError, "identity.*inspection"
            ):
                lock.acquire()

        close.assert_called_once()
        self.assertTrue(self.lock_path.exists())
        self.assertFalse(lock.release())

    def test_partial_lock_writes_are_completed(self):
        lock = self.lock()
        real_write = os.write
        writes = 0

        def partial_write(descriptor: int, data: bytes) -> int:
            nonlocal writes
            writes += 1
            return real_write(descriptor, data[:5])

        with patch(
            "bitguard_bnn.bootstrap.state.os.write",
            side_effect=partial_write,
        ):
            lock.acquire()

        self.assertGreater(writes, 1)
        self.assertTrue(lock.release())

    def test_zero_progress_lock_write_closes_and_cleans_created_lock(self):
        lock = self.lock()
        real_close = os.close

        with (
            patch("bitguard_bnn.bootstrap.state.os.write", return_value=0),
            patch(
                "bitguard_bnn.bootstrap.state.os.close",
                side_effect=real_close,
            ) as close,
        ):
            with self.assertRaisesRegex(BootstrapLockError, "write.*no progress"):
                lock.acquire()

        self.assertGreaterEqual(close.call_count, 1)
        self.assertFalse(self.lock_path.exists())
        self.assertFalse(lock.release())

    def test_lock_fsync_failure_closes_and_cleans_created_lock(self):
        lock = self.lock()
        real_close = os.close

        with (
            patch(
                "bitguard_bnn.bootstrap.state.os.fsync",
                side_effect=OSError("simulated lock fsync failure"),
            ),
            patch(
                "bitguard_bnn.bootstrap.state.os.close",
                side_effect=real_close,
            ) as close,
        ):
            with self.assertRaisesRegex(BootstrapLockError, "fsync"):
                lock.acquire()

        self.assertGreaterEqual(close.call_count, 1)
        self.assertFalse(self.lock_path.exists())
        self.assertFalse(lock.release())

    def test_lock_close_failure_does_not_mark_owned_and_cleans_created_lock(self):
        lock = self.lock()
        real_close = os.close
        failed = False

        def close_then_fail_once(descriptor: int) -> None:
            nonlocal failed
            real_close(descriptor)
            if not failed:
                failed = True
                raise OSError("simulated lock close failure")

        with patch(
            "bitguard_bnn.bootstrap.state.os.close",
            side_effect=close_then_fail_once,
        ):
            with self.assertRaisesRegex(BootstrapLockError, "close"):
                lock.acquire()

        self.assertFalse(self.lock_path.exists())
        self.assertFalse(lock.release())

    def test_lock_cleanup_failure_is_reported_without_raw_oserror(self):
        lock = self.lock()

        with (
            patch(
                "bitguard_bnn.bootstrap.state.os.write",
                side_effect=OSError("simulated lock write failure"),
            ),
            patch.object(
                lock,
                "_discard_created_lock",
                side_effect=BootstrapLockError("simulated safe cleanup failure"),
            ),
        ):
            with self.assertRaisesRegex(
                BootstrapLockError, "write.*safe cleanup.*failed"
            ):
                lock.acquire()

        self.assertTrue(self.lock_path.exists())
        self.assertFalse(lock.release())

    def test_lock_quarantine_close_failure_is_wrapped_as_cleanup_error(self):
        lock = self.lock()
        real_close = os.close
        close_count = 0

        def fail_reservation_close(descriptor: int) -> None:
            nonlocal close_count
            close_count += 1
            real_close(descriptor)
            if close_count == 2:
                raise OSError("simulated quarantine close failure")

        with (
            patch(
                "bitguard_bnn.bootstrap.state.os.write",
                side_effect=OSError("simulated lock write failure"),
            ),
            patch(
                "bitguard_bnn.bootstrap.state.os.close",
                side_effect=fail_reservation_close,
            ),
        ):
            with self.assertRaisesRegex(
                BootstrapLockError, "safe cleanup.*quarantine.*close"
            ):
                lock.acquire()

        self.assertTrue(self.lock_path.exists())
        self.assertFalse(lock.release())

    def test_same_lock_object_can_release_repeatedly_and_reacquire(self):
        lock = self.lock()

        lock.acquire()
        self.assertTrue(lock.release())
        self.assertFalse(lock.release())
        lock.acquire()
        self.assertTrue(lock.release())

    def test_active_same_host_lock_is_refused(self):
        self.write_lock(pid=303)
        liveness = Mock(return_value=True)

        with self.assertRaisesRegex(BootstrapLockError, "active.*PID 303"):
            self.lock(pid_is_alive=liveness, recover_stale=True).acquire()

        liveness.assert_called_once_with(303)
        self.assertTrue(self.lock_path.exists())

    def test_stale_same_host_lock_requires_explicit_recovery(self):
        self.write_lock(pid=404)
        liveness = Mock(return_value=False)

        with self.assertRaisesRegex(BootstrapLockError, "stale.*explicit"):
            self.lock(pid_is_alive=liveness).acquire()

        self.assertTrue(self.lock_path.exists())

    def test_explicit_recovery_rechecks_and_reacquires_with_exclusive_create(self):
        self.write_lock(pid=404)
        liveness = Mock(return_value=False)
        lock = self.lock(pid_is_alive=liveness, recover_stale=True)

        lock.acquire()

        self.assertGreaterEqual(liveness.call_count, 2)
        metadata = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["pid"], 202)
        self.assertEqual(metadata["nonce"], "new-owner")
        self.assertTrue(lock.release())

    def test_stale_recovery_preserves_lock_replaced_during_recheck(self):
        self.write_lock(pid=404, nonce="stale-owner")

        def replace_before_recheck(pid: int) -> bool:
            if pid == 404:
                self.write_lock(pid=505, nonce="replacement-owner")
                return False
            return True

        with self.assertRaisesRegex(BootstrapLockError, "active.*PID 505"):
            self.lock(
                pid_is_alive=replace_before_recheck,
                recover_stale=True,
            ).acquire()

        metadata = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["nonce"], "replacement-owner")

    def test_stale_recovery_restores_late_replacement_instead_of_deleting_it(self):
        self.write_lock(pid=404, nonce="stale-owner")
        winner_path = self.lock_path.with_name("late-winner.lock")
        self.write_lock_at(winner_path, pid=505, nonce="late-winner")
        real_replace = os.replace
        real_unlink = Path.unlink
        injected = False

        def install_winner() -> None:
            nonlocal injected
            if not injected:
                injected = True
                real_replace(winner_path, self.lock_path)

        def replace_after_last_check(source, target) -> None:
            if Path(source) == self.lock_path:
                install_winner()
            real_replace(source, target)

        def unlink_after_last_check(path, *args, **kwargs) -> None:
            if path == self.lock_path:
                install_winner()
            real_unlink(path, *args, **kwargs)

        with (
            patch(
                "bitguard_bnn.bootstrap.state.os.replace",
                side_effect=replace_after_last_check,
            ),
            patch.object(
                Path,
                "unlink",
                autospec=True,
                side_effect=unlink_after_last_check,
            ),
        ):
            with self.assertRaisesRegex(BootstrapLockError, "changed.*recovery"):
                self.lock(
                    pid_is_alive=lambda _pid: False,
                    recover_stale=True,
                ).acquire()

        metadata = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["nonce"], "late-winner")
        self.assertEqual(self.quarantine_paths(), [])

    def test_contender_winning_after_stale_quarantine_is_preserved(self):
        self.write_lock(pid=404, nonce="stale-owner")
        real_open = os.open
        exclusive_attempts = 0
        probed_pids: list[int] = []

        def contender_wins(path, flags, mode=0o777):
            nonlocal exclusive_attempts
            if Path(path) == self.lock_path and flags & os.O_EXCL:
                exclusive_attempts += 1
                if exclusive_attempts == 2:
                    self.write_lock(pid=505, nonce="winning-contender")
            return real_open(path, flags, mode)

        def absent(pid: int) -> bool:
            probed_pids.append(pid)
            return False

        with patch(
            "bitguard_bnn.bootstrap.state.os.open",
            side_effect=contender_wins,
        ):
            with self.assertRaisesRegex(
                BootstrapLockError, "contender.*stale recovery"
            ):
                self.lock(pid_is_alive=absent, recover_stale=True).acquire()

        metadata = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["nonce"], "winning-contender")
        self.assertNotIn(505, probed_pids)
        self.assertEqual(exclusive_attempts, 2)
        self.assertEqual(self.quarantine_paths(), [])

    def test_failed_mismatch_restore_preserves_quarantine_and_new_owner(self):
        self.write_lock(pid=404, nonce="stale-owner")
        winner_path = self.lock_path.with_name("late-winner.lock")
        self.write_lock_at(winner_path, pid=505, nonce="quarantined-winner")
        real_replace = os.replace
        injected = False

        def replace_then_fill_original(source, target) -> None:
            nonlocal injected
            if Path(source) == self.lock_path and not injected:
                injected = True
                real_replace(winner_path, self.lock_path)
                real_replace(self.lock_path, target)
                self.write_lock(pid=606, nonce="new-canonical-owner")
                return
            real_replace(source, target)

        with patch(
            "bitguard_bnn.bootstrap.state.os.replace",
            side_effect=replace_then_fill_original,
        ):
            with self.assertRaisesRegex(
                BootstrapLockError, "quarantine.*preserved"
            ) as caught:
                self.lock(
                    pid_is_alive=lambda _pid: False,
                    recover_stale=True,
                ).acquire()

        canonical = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(canonical["nonce"], "new-canonical-owner")
        quarantines = self.quarantine_paths()
        self.assertEqual(len(quarantines), 1)
        self.assertIn(str(quarantines[0]), str(caught.exception))
        quarantined = json.loads(quarantines[0].read_text(encoding="utf-8"))
        self.assertEqual(quarantined["nonce"], "quarantined-winner")

    def test_old_but_active_lock_is_never_removed_based_on_age(self):
        self.write_lock(pid=606, started_at="2001-01-01T00:00:00Z")

        with self.assertRaisesRegex(BootstrapLockError, "active.*PID 606"):
            self.lock(pid_is_alive=lambda _pid: True, recover_stale=True).acquire()

        self.assertTrue(self.lock_path.exists())

    def test_foreign_host_lock_is_refused_even_with_explicit_recovery(self):
        self.write_lock(hostname="remote-host", pid=707)
        liveness = Mock(side_effect=AssertionError("foreign PID must not be probed"))

        with self.assertRaisesRegex(BootstrapLockError, "different host"):
            self.lock(pid_is_alive=liveness, recover_stale=True).acquire()

        liveness.assert_not_called()
        self.assertTrue(self.lock_path.exists())

    def test_corrupt_lock_fails_closed_with_recovery_guidance(self):
        self.lock_path.write_text("{truncated", encoding="utf-8")

        with self.assertRaisesRegex(BootstrapLockError, "invalid.*manually"):
            self.lock(recover_stale=True).acquire()

        self.assertTrue(self.lock_path.exists())

    def test_release_preserves_replaced_foreign_lock(self):
        lock = self.lock()
        lock.acquire()
        self.write_lock(pid=808, nonce="replacement-owner")

        self.assertFalse(lock.release())

        self.assertTrue(self.lock_path.exists())
        metadata = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["nonce"], "replacement-owner")

    def test_release_preserves_byte_identical_replacement_with_new_identity(self):
        lock = self.lock()
        lock.acquire()
        owned_bytes = self.lock_path.read_bytes()
        owned_identity = (self.lock_path.stat().st_dev, self.lock_path.stat().st_ino)
        replacement = self.lock_path.with_name("identical-replacement.lock")
        replacement.write_bytes(owned_bytes)
        replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
        self.assertNotEqual(replacement_identity, owned_identity)
        os.replace(replacement, self.lock_path)

        self.assertFalse(lock.release())

        self.assertTrue(self.lock_path.exists())
        self.assertEqual(self.lock_path.read_bytes(), owned_bytes)
        self.assertEqual(
            (self.lock_path.stat().st_dev, self.lock_path.stat().st_ino),
            replacement_identity,
        )
        self.assertEqual(self.quarantine_paths(), [])


if __name__ == "__main__":
    unittest.main()
