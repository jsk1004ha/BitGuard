from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bitguard_bnn.bootstrap import fsops
from bitguard_bnn.bootstrap.fsops import RetirementError, retire_owned_path


def _identity(path: Path) -> tuple[int, int]:
    value = path.lstat()
    return (value.st_dev, value.st_ino)


class OwnedPathRetirementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.candidate = self.root / "candidate.bin"

    def test_owned_file_is_moved_to_a_private_recovery_directory(self) -> None:
        payload = b"writer-owned bytes"
        self.candidate.write_bytes(payload)
        expected = _identity(self.candidate)

        result = retire_owned_path(
            self.candidate,
            expected,
            purpose="unit-test cleanup",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(self.candidate.exists())
        self.assertEqual(result.artifact.read_bytes(), payload)
        self.assertEqual(_identity(result.artifact), expected)
        self.assertEqual(result.artifact.parent, result.quarantine)
        self.assertTrue(result.quarantine.name.startswith(".bitguard-retired-"))
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(result.quarantine.stat().st_mode), 0o700)

    def test_reclaim_truncates_only_the_validated_retired_descriptor(self) -> None:
        self.candidate.write_bytes(b"duplicate archive storage")
        expected = _identity(self.candidate)

        result = retire_owned_path(
            self.candidate,
            expected,
            purpose="completed partial cleanup",
            reclaim_storage=True,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.artifact.stat().st_size, 0)
        self.assertTrue(result.storage_reclaimed)

    def test_replacement_between_observation_and_move_is_restored_and_preserved(self) -> None:
        owned = b"owned bytes"
        foreign = b"foreign replacement must survive"
        displaced = self.root / "displaced-owned.bin"
        replacement = self.root / "foreign.bin"
        self.candidate.write_bytes(owned)
        replacement.write_bytes(foreign)
        expected = _identity(self.candidate)
        real_rename = fsops._rename_noreplace
        substituted = False

        def replace_before_move(
            source: object,
            destination: object,
            directory_descriptor: int | None,
        ) -> None:
            nonlocal substituted
            if Path(source) == self.candidate and not substituted:
                substituted = True
                self.candidate.replace(displaced)
                replacement.replace(self.candidate)
            real_rename(Path(source), Path(destination), directory_descriptor)

        with patch.object(fsops, "_rename_noreplace", side_effect=replace_before_move):
            with self.assertRaisesRegex(RetirementError, "foreign|recovery|restored") as caught:
                retire_owned_path(
                    self.candidate,
                    expected,
                    purpose="adversarial cleanup",
                )

        self.assertTrue(substituted)
        self.assertEqual(displaced.read_bytes(), owned)
        self.assertEqual(self.candidate.read_bytes(), foreign)
        self.assertIsNotNone(caught.exception.recovery_path)
        assert caught.exception.recovery_path is not None
        self.assertEqual(caught.exception.recovery_path.read_bytes(), foreign)

    def test_destination_collision_is_preserved_and_reported(self) -> None:
        self.candidate.write_bytes(b"owned")
        expected = _identity(self.candidate)
        collision = b"existing quarantine child"

        def collide_before_move(
            _source: object,
            destination: object,
            _directory_descriptor: int | None,
        ) -> None:
            Path(destination).write_bytes(collision)
            raise FileExistsError(str(destination))

        with patch.object(fsops, "_rename_noreplace", side_effect=collide_before_move):
            with self.assertRaisesRegex(RetirementError, "collision|recovery") as caught:
                retire_owned_path(
                    self.candidate,
                    expected,
                    purpose="collision cleanup",
                )

        self.assertEqual(self.candidate.read_bytes(), b"owned")
        self.assertIsNotNone(caught.exception.recovery_path)
        assert caught.exception.recovery_path is not None
        self.assertEqual(caught.exception.recovery_path.read_bytes(), collision)

    def test_boolean_and_malformed_arguments_are_rejected(self) -> None:
        self.candidate.write_bytes(b"owned")
        expected = _identity(self.candidate)

        with self.assertRaisesRegex(TypeError, "reclaim_storage"):
            retire_owned_path(
                self.candidate,
                expected,
                purpose="invalid options",
                reclaim_storage=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "identity"):
            retire_owned_path(
                self.candidate,
                (True, expected[1]),
                purpose="invalid identity",
            )
        with self.assertRaisesRegex(ValueError, "purpose"):
            retire_owned_path(self.candidate, expected, purpose="")

    def test_reclaim_never_truncates_a_replaced_quarantine_name(self) -> None:
        owned = b"owned archive bytes must not be truncated after displacement"
        foreign = b"foreign quarantine replacement must survive"
        displaced = self.root / "displaced-retired-artifact"
        replacement = self.root / "foreign-retired-artifact"
        self.candidate.write_bytes(owned)
        replacement.write_bytes(foreign)
        expected = _identity(self.candidate)
        real_open = fsops.os.open
        substituted = False

        def replace_before_descriptor_validation(
            path: object,
            flags: int,
            mode: int = 0o777,
        ) -> int:
            nonlocal substituted
            candidate = Path(path)
            if candidate.name == "artifact" and flags & os.O_WRONLY and not substituted:
                substituted = True
                candidate.replace(displaced)
                replacement.replace(candidate)
            return real_open(path, flags, mode)

        with patch.object(
            fsops.os,
            "open",
            side_effect=replace_before_descriptor_validation,
        ):
            with self.assertRaisesRegex(RetirementError, "changed|foreign|preserved"):
                retire_owned_path(
                    self.candidate,
                    expected,
                    purpose="completed partial cleanup",
                    reclaim_storage=True,
                )

        self.assertTrue(substituted)
        self.assertEqual(displaced.read_bytes(), owned)
        quarantine_artifacts = list(self.root.glob(".bitguard-retired-*/artifact"))
        self.assertEqual(len(quarantine_artifacts), 1)
        self.assertEqual(quarantine_artifacts[0].read_bytes(), foreign)

    def test_reclaim_os_errors_are_typed_and_keep_the_recovery_artifact(self) -> None:
        payload = b"reclaim failure must retain these bytes"
        self.candidate.write_bytes(payload)
        expected = _identity(self.candidate)

        with patch.object(
            fsops.os,
            "ftruncate",
            side_effect=OSError("injected truncate failure"),
        ):
            with self.assertRaisesRegex(RetirementError, "reclaim|preserved") as caught:
                retire_owned_path(
                    self.candidate,
                    expected,
                    purpose="completed partial cleanup",
                    reclaim_storage=True,
                )

        self.assertIsNotNone(caught.exception.recovery_path)
        assert caught.exception.recovery_path is not None
        self.assertEqual(caught.exception.recovery_path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
