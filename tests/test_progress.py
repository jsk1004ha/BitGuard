from __future__ import annotations

import unittest
from io import StringIO

from bitguard_bnn.progress import TerminalProgress


class _TTY(StringIO):
    def isatty(self) -> bool:
        return True


class _NotTTY(StringIO):
    def isatty(self) -> bool:
        return False


class TerminalProgressTest(unittest.TestCase):
    def test_inspection_progress_truncates_long_windows_path_to_line_width(self) -> None:
        stream = _TTY()
        progress = TerminalProgress(
            stream=stream,
            min_interval=0.0,
            max_line_width=80,
        )

        progress(
            {
                "scope": "bootstrap",
                "status": "advanced",
                "stage": "inspect",
                "completed": 4,
                "total": 9,
                "dataset": "nbaiot",
                "phase": "validate",
                "relative_path": (
                    "device_with_a_very_long_windows_directory_name\\"
                    "nested_attack_capture_directory\\"
                    "benign_traffic_capture.csv"
                ),
                "file_index": 2,
                "file_count": 10,
                "rows": 25_000,
            }
        )

        updates = [
            update.split("\n", 1)[0]
            for update in stream.getvalue().split("\r")
            if update
        ]
        self.assertTrue(updates)
        self.assertTrue(all(len(update) <= 80 for update in updates))
        self.assertIn("nbaiot", updates[-1])
        self.assertIn("validate", updates[-1])
        self.assertIn("2/10", updates[-1])
        self.assertIn("25,000 rows", updates[-1])
        self.assertIn(".csv", updates[-1])

    def test_inspection_progress_shows_dataset_phase_file_and_rows(self) -> None:
        stream = _TTY()
        progress = TerminalProgress(stream=stream, min_interval=0.0)

        progress(
            {
                "scope": "bootstrap",
                "status": "advanced",
                "stage": "inspect",
                "completed": 4,
                "total": 9,
                "dataset": "nbaiot",
                "phase": "validate",
                "relative_path": "device/benign_traffic.csv",
                "file_index": 2,
                "file_count": 10,
                "rows": 25_000,
            }
        )

        rendered = stream.getvalue()
        self.assertIn("44%", rendered)
        self.assertIn("nbaiot", rendered)
        self.assertIn("validate", rendered)
        self.assertIn("device/benign_traffic.csv", rendered)
        self.assertIn("2/10", rendered)
        self.assertIn("25,000 rows", rendered)

    def test_bootstrap_stage_progress_reaches_one_hundred_percent(self) -> None:
        stream = _TTY()
        progress = TerminalProgress(stream=stream, min_interval=0.0)

        progress(
            {
                "scope": "bootstrap",
                "status": "started",
                "stage": "acquire",
                "completed": 2,
                "total": 9,
            }
        )
        progress(
            {
                "scope": "bootstrap",
                "status": "completed",
                "stage": "summarize",
                "completed": 9,
                "total": 9,
            }
        )

        rendered = stream.getvalue()
        self.assertIn("acquire", rendered)
        self.assertIn("22%", rendered)
        self.assertIn("100%", rendered)
        self.assertTrue(rendered.endswith("\n"))

    def test_training_progress_shows_dataset_role_epoch_and_rows(self) -> None:
        stream = _TTY()
        progress = TerminalProgress(stream=stream, min_interval=0.0)

        progress(
            {
                "scope": "training",
                "status": "advanced",
                "dataset": "nbaiot",
                "role": "main",
                "epoch": 3,
                "epochs": 10,
                "completed": 50,
                "total": 100,
            }
        )

        rendered = stream.getvalue()
        self.assertIn("nbaiot/main", rendered)
        self.assertIn("epoch 3/10", rendered)
        self.assertIn("50%", rendered)

    def test_non_tty_stream_is_left_clean(self) -> None:
        stream = _NotTTY()
        progress = TerminalProgress(stream=stream)

        progress(
            {
                "scope": "bootstrap",
                "status": "completed",
                "stage": "inspect",
                "completed": 5,
                "total": 9,
            }
        )

        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
