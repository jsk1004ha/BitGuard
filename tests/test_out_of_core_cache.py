from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import bitguard_bnn.out_of_core.cache as cache_module
from bitguard_bnn.out_of_core.cache import (
    CacheLayout,
    CalibrationCache,
    class_weights_from_counts,
)
from bitguard_bnn.preprocess import class_weights


class ClassWeightTests(unittest.TestCase):
    def test_counts_match_existing_array_implementation(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 2, 2], dtype=np.int64)
        actual = class_weights_from_counts(
            {"benign": 3, "scan": 1, "dos": 2},
            ("benign", "scan", "dos"),
        )
        np.testing.assert_array_equal(actual, class_weights(labels, 3))

    def test_invalid_counts_fail_closed(self) -> None:
        invalid = (
            ({"benign": 1}, ("benign", "scan")),
            ({"benign": 1, "scan": 0}, ("benign", "scan")),
            ({"benign": 1, "scan": -1}, ("benign", "scan")),
            ({"benign": 1, "scan": True}, ("benign", "scan")),
            ({"benign": 1, "scan": 1.0}, ("benign", "scan")),
        )
        for counts, labels in invalid:
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    class_weights_from_counts(counts, labels)


class CalibrationCacheTests(unittest.TestCase):
    def _layout(self, **overrides: object) -> CacheLayout:
        values: dict[str, object] = {
            "prepared_descriptor_fingerprint": "prepared",
            "shard_fingerprint": "shards",
            "preprocessor_fingerprint": "preprocessor",
            "source_fingerprint": "source",
            "split": "validation",
            "row_count": 5,
            "class_labels": ("benign", "attack"),
            "selected_features": ("f1", "f2"),
            "boolean_features": ("flag",),
            "device_id_width": 12,
            "source_id_width": 16,
        }
        values.update(overrides)
        return CacheLayout(**values)  # type: ignore[arg-type]

    def _batch(self, start: int, rows: int) -> dict[str, object]:
        known = np.tile(np.asarray([[0.75, 0.25]], dtype=np.float32), (rows, 1))
        return {
            "cache_position": np.arange(start, start + rows, dtype=np.int64),
            "uid_digest": np.arange(rows * 32, dtype=np.uint8).reshape(rows, 32),
            "true_label": np.arange(rows, dtype=np.int32) % 2,
            "known_probabilities": known,
            "selected_values": np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
            "tiny_benign_probability": np.full(rows, 0.6, dtype=np.float32),
            "boolean_flags": np.zeros((rows, 1), dtype=np.bool_),
            "timestamp": np.arange(start, start + rows, dtype=np.float64) + 0.5,
            "sequence": np.arange(start, start + rows, dtype=np.int64),
            "device_id": [f"기기-{index}" for index in range(start, start + rows)],
            "source_id": [f"source\x00{index}" for index in range(start, start + rows)],
            "routed_probabilities": known.copy(),
            "exit_stage": np.zeros(rows, dtype=np.int16),
        }

    def test_create_preallocates_exact_shapes_and_commits_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = CalibrationCache.create(Path(temporary) / "cache", self._layout())
            self.assertEqual(cache.committed_rows, 0)
            self.assertEqual(cache.arrays["uid_digest"].shape, (5, 32))
            self.assertEqual(cache.arrays["known_probabilities"].shape, (5, 2))
            self.assertEqual(cache.arrays["selected_values"].shape, (5, 2))
            self.assertEqual(cache.arrays["boolean_flags"].shape, (5, 1))
            for name, array in cache.arrays.items():
                self.assertEqual(
                    (cache.root / f"{name}.bin").stat().st_size,
                    array.nbytes,
                )
            cache.close()

    def test_multiple_commits_reopen_readonly_and_preserve_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            with CalibrationCache.create(root, layout) as cache:
                cache.commit_range(0, self._batch(0, 2))
                cache.commit_range(2, self._batch(2, 3))
                self.assertEqual(cache.committed_rows, 5)
            with CalibrationCache.open_readonly(root, layout) as cache:
                self.assertTrue(cache.readonly)
                np.testing.assert_array_equal(
                    cache.arrays["cache_position"], np.arange(5, dtype=np.int64)
                )
                self.assertEqual(cache.read_identifiers("device_id", 0, 2), ("기기-0", "기기-1"))
                self.assertEqual(cache.read_identifiers("source_id", 0, 2), ("source\x000", "source\x001"))
                with self.assertRaises(ValueError):
                    cache.arrays["true_label"][0] = 1

    def test_failed_journal_advance_reuses_only_previous_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            cache = CalibrationCache.create(root, layout)
            cache.commit_range(0, self._batch(0, 2))
            with mock.patch.object(
                cache_module, "_write_journal", side_effect=OSError("injected")
            ):
                with self.assertRaises(OSError):
                    cache.commit_range(2, self._batch(2, 2))
            self.assertEqual(cache.committed_rows, 2)
            cache.close()
            with CalibrationCache.open_resume(root, layout) as resumed:
                self.assertEqual(resumed.committed_rows, 2)
                resumed.commit_range(2, self._batch(2, 3))
                self.assertEqual(resumed.committed_rows, 5)

    def test_mismatch_tamper_and_truncation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            with CalibrationCache.create(root, layout) as cache:
                cache.commit_range(0, self._batch(0, 2))
            with self.assertRaises(RuntimeError):
                CalibrationCache.open_readonly(
                    root, self._layout(source_fingerprint="different")
                )

            known_path = root / "known_probabilities.bin"
            with known_path.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"\xff")
            with self.assertRaisesRegex(RuntimeError, "committed range fingerprint"):
                CalibrationCache.open_readonly(root, layout)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            with CalibrationCache.create(root, layout):
                pass
            uid_path = root / "uid_digest.bin"
            with uid_path.open("r+b") as handle:
                handle.truncate(uid_path.stat().st_size - 1)
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                CalibrationCache.open_readonly(root, layout)

    def test_invalid_ranges_data_and_identifier_overflow_do_not_advance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = CalibrationCache.create(Path(temporary) / "cache", self._layout())
            invalid_batches: list[tuple[int, dict[str, object]]] = []
            wrong_position = self._batch(0, 2)
            wrong_position["cache_position"] = np.asarray([0, 2], dtype=np.int64)
            invalid_batches.append((0, wrong_position))
            wrong_dtype = self._batch(0, 2)
            wrong_dtype["true_label"] = np.asarray([0, 1], dtype=np.int64)
            invalid_batches.append((0, wrong_dtype))
            nonfinite = self._batch(0, 2)
            nonfinite["selected_values"] = np.asarray(
                [[0.0, np.nan], [1.0, 2.0]], dtype=np.float32
            )
            invalid_batches.append((0, nonfinite))
            overflow = self._batch(0, 2)
            overflow["device_id"] = ["x" * 13, "ok"]
            invalid_batches.append((0, overflow))
            missing = self._batch(0, 2)
            missing.pop("sequence")
            invalid_batches.append((0, missing))
            for start, batch in invalid_batches:
                with self.subTest(keys=tuple(batch)):
                    with self.assertRaises(ValueError):
                        cache.commit_range(start, batch)
                    self.assertEqual(cache.committed_rows, 0)
            with self.assertRaises(ValueError):
                cache.commit_range(1, self._batch(1, 1))
            cache.close()

    def test_close_releases_windows_file_handles_for_recursive_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            cache = CalibrationCache.create(root, self._layout())
            cache.close()
            for artifact in root.iterdir():
                artifact.unlink()
            root.rmdir()
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
