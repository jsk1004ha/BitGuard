from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

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


class _FailingTextStream:
    def __init__(self, stream, operation: str) -> None:
        self._stream = stream
        self._operation = operation

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._stream.__exit__(exc_type, exc, traceback)

    def write(self, value: str) -> int:
        if self._operation == "write":
            raise OSError("simulated state write failure")
        return self._stream.write(value)

    def flush(self) -> None:
        if self._operation == "flush":
            raise OSError("simulated state flush failure")
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()


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

    def test_persistence_flushes_and_atomically_replaces_sibling_temp_file(self):
        output = self.output("archive.zip")
        store = BootstrapStateStore(self.state_path)

        with (
            patch(
                "bitguard_bnn.bootstrap.state.Path.replace",
                autospec=True,
                side_effect=lambda source, target: os.replace(source, target),
            ) as replace,
            patch(
                "bitguard_bnn.bootstrap.state.os.fsync",
                wraps=os.fsync,
            ) as fsync,
        ):
            store.complete("acquire", "input-a", [output])

        replace.assert_called_once_with(self.root / "state.json.tmp", self.state_path)
        fsync.assert_called_once()
        self.assertFalse((self.root / "state.json.tmp").exists())

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
        self.assertFalse((self.root / "state.json.tmp").exists())

    def _assert_temp_stream_failure_preserves_previous_state(
        self, operation: str
    ) -> None:
        archive = self.output("archive.zip")
        inspection = self.output("inspection.json")
        store = BootstrapStateStore(self.state_path)
        store.complete("acquire", "input-a", [archive])
        previous_bytes = self.state_path.read_bytes()
        temporary = self.root / "state.json.tmp"
        real_open = Path.open

        def open_with_failure(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            if path == temporary:
                return _FailingTextStream(stream, operation)
            return stream

        with patch.object(
            Path,
            "open",
            autospec=True,
            side_effect=open_with_failure,
        ):
            with self.assertRaisesRegex(BootstrapStateError, "persist"):
                store.complete("inspect", "input-b", [inspection])

        self.assertEqual(self.state_path.read_bytes(), previous_bytes)
        self.assertEqual(store.completed_stages, ("acquire",))
        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages, ("acquire",)
        )
        self.assertFalse(temporary.exists())

    def test_failed_state_write_preserves_previous_state_and_cleans_temp(self):
        self._assert_temp_stream_failure_preserves_previous_state("write")

    def test_failed_state_flush_preserves_previous_state_and_cleans_temp(self):
        self._assert_temp_stream_failure_preserves_previous_state("flush")

    def test_failed_state_fsync_preserves_previous_state_and_cleans_temp(self):
        archive = self.output("archive.zip")
        inspection = self.output("inspection.json")
        store = BootstrapStateStore(self.state_path)
        store.complete("acquire", "input-a", [archive])
        previous_bytes = self.state_path.read_bytes()

        with patch(
            "bitguard_bnn.bootstrap.state.os.fsync",
            side_effect=OSError("simulated state fsync failure"),
        ):
            with self.assertRaisesRegex(BootstrapStateError, "persist"):
                store.complete("inspect", "input-b", [inspection])

        self.assertEqual(self.state_path.read_bytes(), previous_bytes)
        self.assertEqual(store.completed_stages, ("acquire",))
        self.assertEqual(
            BootstrapStateStore(self.state_path).completed_stages, ("acquire",)
        )
        self.assertFalse((self.root / "state.json.tmp").exists())

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
