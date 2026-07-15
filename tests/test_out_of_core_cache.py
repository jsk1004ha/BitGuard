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
                    class_weights_from_counts(counts, labels)  # type: ignore[arg-type]


class CalibrationCacheTests(unittest.TestCase):
    def _layout(self, **overrides: object) -> CacheLayout:
        values: dict[str, object] = {
            "prepared_descriptor_fingerprint": "prepared",
            "shard_fingerprint": "shards",
            "preprocessor_fingerprint": "preprocessor",
            "source_fingerprint": "source",
            "main_checkpoint_fingerprint": "main-checkpoint",
            "tiny_checkpoint_fingerprint": "tiny-checkpoint",
            "inference_contract_fingerprint": "inference-contract",
            "split": "validation",
            "row_count": 5,
            "main_class_labels": ("benign", "scan_like"),
            "routed_class_labels": ("benign", "scan_like", "unknown_like"),
            "true_class_labels": ("benign", "scan_like", "unknown_like"),
            "selected_features": ("f1", "f2"),
            "boolean_features": ("flag",),
            "device_id_width": 12,
            "source_id_width": 16,
        }
        values.update(overrides)
        return CacheLayout(**values)  # type: ignore[arg-type]

    def _inference_batch(self, start: int, rows: int) -> dict[str, object]:
        return {
            "cache_position": np.arange(start, start + rows, dtype=np.int64),
            "uid_digest": np.arange(rows * 32, dtype=np.uint8).reshape(rows, 32),
            "true_label": np.arange(rows, dtype=np.int32) % 2,
            "known_probabilities": np.tile(
                np.asarray([[0.75, 0.25]], dtype=np.float32), (rows, 1)
            ),
            "selected_values": np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
            "tiny_benign_probability": np.full(rows, 0.6, dtype=np.float32),
            "boolean_flags": np.zeros((rows, 1), dtype=np.bool_),
            "timestamp": np.arange(start, start + rows, dtype=np.float64) + 0.5,
            "sequence": np.arange(start, start + rows, dtype=np.int64),
            "device_id": [f"기기-{index}" for index in range(start, start + rows)],
            "source_id": [f"source\x00{index}" for index in range(start, start + rows)],
        }

    def _routing_batch(self, rows: int) -> dict[str, object]:
        return {
            "routed_probabilities": np.tile(
                np.asarray([[0.70, 0.20, 0.10]], dtype=np.float32), (rows, 1)
            ),
            "exit_stage": np.zeros(rows, dtype=np.int16),
        }

    def _commit_full_inference(self, cache: CalibrationCache) -> None:
        cache.commit_inference_range(0, self._inference_batch(0, 2))
        cache.commit_inference_range(2, self._inference_batch(2, 3))

    def test_create_preallocates_two_empty_phases_with_exact_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = CalibrationCache.create(Path(temporary) / "cache", self._layout())
            self.assertEqual(cache.committed_rows, 0)
            self.assertEqual(cache.routed_committed_rows, 0)
            self.assertEqual(cache.arrays["uid_digest"].shape, (5, 32))
            self.assertEqual(cache.arrays["known_probabilities"].shape, (5, 2))
            self.assertEqual(cache.arrays["routed_probabilities"].shape, (5, 3))
            for name, array in cache.arrays.items():
                self.assertEqual((cache.root / f"{name}.bin").stat().st_size, array.nbytes)
            cache.close()

    def test_layout_binds_label_model_and_inference_identities(self) -> None:
        baseline = self._layout()
        variants = (
            self._layout(main_class_labels=("benign", "flood_like")),
            self._layout(routed_class_labels=("benign", "scan_like", "unknown_like", "x")),
            self._layout(true_class_labels=("benign", "unknown_like")),
            self._layout(main_checkpoint_fingerprint="other-main"),
            self._layout(tiny_checkpoint_fingerprint=None),
            self._layout(inference_contract_fingerprint="other-inference"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            with CalibrationCache.create(root, baseline):
                pass
            for variant in variants:
                with self.subTest(variant=variant):
                    self.assertNotEqual(baseline.fingerprint, variant.fingerprint)
                    with self.assertRaisesRegex(RuntimeError, "expected layout"):
                        CalibrationCache.open_readonly(root, variant)
        with self.assertRaisesRegex(ValueError, "not supported"):
            self._layout(algorithm="bitguard.calibration-cache.future")

    def test_inference_then_routing_commit_and_readonly_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            with CalibrationCache.create(root, layout) as cache:
                self._commit_full_inference(cache)
                self.assertEqual(cache.committed_rows, 5)
                self.assertEqual(cache.routed_committed_rows, 0)
                cache.commit_routing_range(
                    0, self._routing_batch(2), routing_contract_fingerprint="routing-v1"
                )
                cache.commit_routing_range(
                    2, self._routing_batch(3), routing_contract_fingerprint="routing-v1"
                )
            with CalibrationCache.open_readonly(
                root, layout, expected_routing_contract_fingerprint="routing-v1"
            ) as cache:
                self.assertEqual(cache.committed_rows, 5)
                self.assertEqual(cache.routed_committed_rows, 5)
                self.assertEqual(cache.routing_contract_fingerprint, "routing-v1")
                self.assertEqual(cache.read_identifiers("device_id", 0, 2), ("기기-0", "기기-1"))
                with self.assertRaises(ValueError):
                    cache.arrays["true_label"][0] = 1
                with self.assertRaises(RuntimeError):
                    cache.commit_inference_range(5, self._inference_batch(5, 1))

    def test_routing_requires_complete_inference_and_matching_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            with CalibrationCache.create(root, layout) as cache:
                cache.commit_inference_range(0, self._inference_batch(0, 2))
                with self.assertRaisesRegex(ValueError, "complete inference"):
                    cache.commit_routing_range(
                        0, self._routing_batch(2), routing_contract_fingerprint="routing-v1"
                    )
                cache.commit_inference_range(2, self._inference_batch(2, 3))
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    cache.commit_routing_range(
                        0, self._routing_batch(2), routing_contract_fingerprint=""
                    )
                invalid_routing = self._routing_batch(2)
                invalid_routing["routed_probabilities"] = np.asarray(
                    [[0.9, 0.2, -0.1], [0.7, 0.2, 0.1]], dtype=np.float32
                )
                with self.assertRaises(ValueError):
                    cache.commit_routing_range(
                        0,
                        invalid_routing,
                        routing_contract_fingerprint="routing-v1",
                    )
                cache.commit_routing_range(
                    0, self._routing_batch(2), routing_contract_fingerprint="routing-v1"
                )
                with self.assertRaisesRegex(ValueError, "routing contract"):
                    cache.commit_routing_range(
                        2, self._routing_batch(3), routing_contract_fingerprint="routing-v2"
                    )
            with self.assertRaisesRegex(RuntimeError, "routing contract"):
                CalibrationCache.open_resume(
                    root, layout, expected_routing_contract_fingerprint="routing-v2"
                )
            with self.assertRaisesRegex(RuntimeError, "expected routing contract"):
                CalibrationCache.open_resume(root, layout)

    def test_inference_journal_failure_before_replace_requires_old_prefix_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            cache = CalibrationCache.create(root, layout)
            cache.commit_inference_range(0, self._inference_batch(0, 2))
            with mock.patch.object(cache_module, "_write_journal", side_effect=OSError("before")):
                with self.assertRaisesRegex(OSError, "before"):
                    cache.commit_inference_range(2, self._inference_batch(2, 2))
            with self.assertRaisesRegex(RuntimeError, "unusable"):
                _ = cache.committed_rows
            with CalibrationCache.open_resume(root, layout) as resumed:
                self.assertEqual(resumed.committed_rows, 2)

    def test_inference_journal_failure_after_replace_reopens_new_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            cache = CalibrationCache.create(root, layout)
            with mock.patch.object(cache_module, "_fsync_directory", side_effect=OSError("after")):
                with self.assertRaisesRegex(OSError, "after"):
                    cache.commit_inference_range(0, self._inference_batch(0, 2))
            with self.assertRaisesRegex(RuntimeError, "unusable"):
                _ = cache.arrays
            with CalibrationCache.open_resume(root, layout) as resumed:
                self.assertEqual(resumed.committed_rows, 2)
                self.assertEqual(resumed.routed_committed_rows, 0)

    def test_routing_journal_failure_before_replace_requires_old_prefix_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            cache = CalibrationCache.create(root, layout)
            self._commit_full_inference(cache)
            with mock.patch.object(cache_module, "_write_journal", side_effect=OSError("before")):
                with self.assertRaisesRegex(OSError, "before"):
                    cache.commit_routing_range(
                        0, self._routing_batch(2), routing_contract_fingerprint="routing-v1"
                    )
            with self.assertRaisesRegex(RuntimeError, "unusable"):
                _ = cache.routed_committed_rows
            with CalibrationCache.open_resume(root, layout) as resumed:
                self.assertEqual(resumed.committed_rows, 5)
                self.assertEqual(resumed.routed_committed_rows, 0)
                self.assertIsNone(resumed.routing_contract_fingerprint)

    def test_routing_journal_failure_after_replace_reopens_new_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            cache = CalibrationCache.create(root, layout)
            self._commit_full_inference(cache)
            with mock.patch.object(cache_module, "_fsync_directory", side_effect=OSError("after")):
                with self.assertRaisesRegex(OSError, "after"):
                    cache.commit_routing_range(
                        0, self._routing_batch(2), routing_contract_fingerprint="routing-v1"
                    )
            with self.assertRaisesRegex(RuntimeError, "unusable"):
                _ = cache.arrays
            with CalibrationCache.open_resume(
                root, layout, expected_routing_contract_fingerprint="routing-v1"
            ) as resumed:
                self.assertEqual(resumed.routed_committed_rows, 2)
                np.testing.assert_allclose(
                    resumed.arrays["routed_probabilities"][:2],
                    self._routing_batch(2)["routed_probabilities"],
                )

    def test_mismatch_tamper_and_truncation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            layout = self._layout()
            with CalibrationCache.create(root, layout) as cache:
                cache.commit_inference_range(0, self._inference_batch(0, 2))
            known_path = root / "known_probabilities.bin"
            with known_path.open("r+b") as handle:
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

    def test_invalid_inference_data_do_not_advance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = CalibrationCache.create(Path(temporary) / "cache", self._layout())
            invalid: list[dict[str, object]] = []
            wrong_position = self._inference_batch(0, 2)
            wrong_position["cache_position"] = np.asarray([0, 2], dtype=np.int64)
            invalid.append(wrong_position)
            wrong_truth = self._inference_batch(0, 2)
            wrong_truth["true_label"] = np.asarray([0, 3], dtype=np.int32)
            invalid.append(wrong_truth)
            nonfinite = self._inference_batch(0, 2)
            nonfinite["selected_values"] = np.asarray(
                [[0.0, np.nan], [1.0, 2.0]], dtype=np.float32
            )
            invalid.append(nonfinite)
            overflow = self._inference_batch(0, 2)
            overflow["device_id"] = ["x" * 13, "ok"]
            invalid.append(overflow)
            missing = self._inference_batch(0, 2)
            missing.pop("sequence")
            invalid.append(missing)
            for batch in invalid:
                with self.assertRaises(ValueError):
                    cache.commit_inference_range(0, batch)
                self.assertEqual(cache.committed_rows, 0)
            with self.assertRaises(ValueError):
                cache.commit_inference_range(0.0, self._inference_batch(0, 1))  # type: ignore[arg-type]
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
