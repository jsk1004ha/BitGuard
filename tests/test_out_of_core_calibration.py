from __future__ import annotations

import copy
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from bitguard_bnn.cascade import (
    apply_boolean_fast_path,
    route_with_temporal_state,
    tune_boolean_fast_path,
    tune_exit_threshold,
)
from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.metrics import calibrate_fixed_fpr_thresholds
from bitguard_bnn.out_of_core.cache import CacheLayout, CalibrationCache
from bitguard_bnn.out_of_core.calibrate import (
    calibrate_fixed_fpr_from_cache,
    calibrate_open_set_from_cache,
    populate_validation_cache,
    route_validation_cache,
    tune_boolean_fast_path_streaming,
    tune_exit_threshold_from_cache,
    validation_contract,
)
from bitguard_bnn.preprocess import FeaturePreprocessor


LABELS = ["benign", "scan_like", "unknown_like"]


class OutOfCoreCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = copy.deepcopy(DEFAULTS)
        self.config["preprocess"]["open_set"].update(
            {
                "enabled": True,
                "benign_distance_quantile": 0.75,
                "max_known_false_unknown_rate": 0.25,
            }
        )
        self.config["cascade"].update(
            {
                "enabled": True,
                "min_attack_recall": 0.75,
                "threshold_grid_size": 31,
                "false_negative_cost": 0.05,
            }
        )

    def _preprocessor(self) -> FeaturePreprocessor:
        processor = FeaturePreprocessor(self.config)
        processor.active_labels = ["benign", "scan_like"]
        processor.label_to_index = {"benign": 0, "scan_like": 1}
        processor.selected_features = ["f1", "f2"]
        processor.benign_center = np.asarray([0.0, 0.0], dtype=np.float32)
        processor.open_distance_threshold = 1.0
        processor.fitted = True
        return processor

    def _layout(self, rows: int) -> CacheLayout:
        return CacheLayout(
            prepared_descriptor_fingerprint="prepared",
            shard_fingerprint="shards",
            preprocessor_fingerprint="preprocessor",
            source_fingerprint="source",
            main_checkpoint_fingerprint="main",
            tiny_checkpoint_fingerprint="tiny",
            inference_contract_fingerprint="inference",
            split="validation",
            row_count=rows,
            main_class_labels=("benign", "scan_like"),
            routed_class_labels=tuple(LABELS),
            true_class_labels=tuple(LABELS),
            selected_features=("f1", "f2"),
            boolean_features=("flag_a", "flag_b"),
            device_id_width=8,
            source_id_width=8,
        )

    def _commit(
        self,
        cache: CalibrationCache,
        labels: np.ndarray,
        known: np.ndarray,
        selected: np.ndarray,
        tiny: np.ndarray,
        flags: np.ndarray,
        timestamps: np.ndarray,
        devices: np.ndarray,
        *,
        batch_rows: int = 3,
        uids: np.ndarray | None = None,
    ) -> None:
        rows = len(labels)
        for start in range(0, rows, batch_rows):
            end = min(start + batch_rows, rows)
            batch_uids = (
                [f"{index + 1:064x}" for index in range(start, end)]
                if uids is None
                else uids[start:end].astype(str).tolist()
            )
            cache.commit_inference_range(
                start,
                {
                    "cache_position": np.arange(start, end, dtype=np.int64),
                    "uid_digest": np.vstack(
                        [np.frombuffer(bytes.fromhex(uid), dtype=np.uint8) for uid in batch_uids]
                    ),
                    "true_label": labels[start:end].astype(np.int32),
                    "known_probabilities": known[start:end].astype(np.float32),
                    "selected_values": selected[start:end].astype(np.float32),
                    "tiny_benign_probability": tiny[start:end].astype(np.float32),
                    "boolean_flags": flags[start:end].astype(np.bool_),
                    "timestamp": timestamps[start:end].astype(np.float64),
                    "sequence": np.arange(start, end, dtype=np.int64),
                    "device_id": devices[start:end].astype(str).tolist(),
                    "source_id": ["capture"] * (end - start),
                },
            )

    def test_open_set_and_tiny_thresholds_match_in_memory(self) -> None:
        labels = np.asarray([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.int32)
        known = np.asarray(
            [[.97, .03], [.78, .22], [.41, .59], [.58, .42],
             [.35, .65], [.20, .80], [.88, .12], [.49, .51]],
            dtype=np.float32,
        )
        selected = np.asarray(
            [[.1, .1], [.4, .2], [1.8, 1.2], [.8, .6],
             [2.0, .8], [2.2, 1.0], [.2, .3], [1.6, .9]],
            dtype=np.float32,
        )
        tiny = np.asarray([.99, .92, .83, .81, .72, .61, .95, .40], dtype=np.float32)
        flags = np.zeros((len(labels), 2), dtype=np.bool_)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with CalibrationCache.create(root / "cache", self._layout(len(labels))) as cache:
                self._commit(
                    cache, labels, known, selected, tiny, flags,
                    np.arange(len(labels), dtype=np.float64),
                    np.full(len(labels), "d", dtype="<U1"),
                )
                processor = self._preprocessor()
                expected = self._preprocessor()
                benign_distance = expected.anomaly_distance(selected[labels == 0])
                expected.open_distance_threshold = max(
                    float(np.quantile(benign_distance, 0.75)), 1e-6
                )
                expected_confidence = expected.calibrate_confidence_threshold(known, selected)
                distance_threshold, confidence_threshold = calibrate_open_set_from_cache(
                    cache, processor, root / "open", chunk_rows=3
                )
                self.assertEqual(distance_threshold, expected.open_distance_threshold)
                self.assertEqual(confidence_threshold, expected_confidence)

                names = np.asarray(LABELS, dtype=str)[labels]
                expected_exit = tune_exit_threshold(tiny, names, 0.75, 31, 0.05)
                actual_exit = tune_exit_threshold_from_cache(
                    cache, 0.75, 31, 0.05, root / "tiny", chunk_rows=3
                )
                self.assertEqual(actual_exit.to_dict(), expected_exit.to_dict())

    def test_validation_inference_cache_resumes_partial_batch_exactly(self) -> None:
        rows = 5
        layout = self._layout(rows)
        inference_contract = validation_contract(
            {"name": "inference"},
            cache_base_fingerprint=layout.inference_base_fingerprint,
            config=self.config,
            tiny_indices=np.asarray([0]),
            encoded_input_dimension=3,
        )
        layout = replace(
            layout, inference_contract_fingerprint=inference_contract
        )
        self.assertEqual(
            layout.inference_base_fingerprint,
            inference_contract.split(".", 1)[0],
        )
        features = np.asarray(
            [[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [1., .2, .4], [.3, 1., .2]],
            dtype=np.float32,
        )
        unencoded = features[:, :2].copy()
        raw_a = np.asarray([.1, 2., .9, 3., 1.1], dtype=np.float32)
        raw_b = np.asarray([.2, 2., .8, 3., .9], dtype=np.float32)
        behavior = np.asarray(["benign", "scan_like", "benign", "scan_like", "benign"])
        uids = np.asarray([f"{index + 11:064x}" for index in range(rows)])

        def batches():
            for start, end in ((0, 3), (3, 5)):
                yield {
                    "features": features[start:end],
                    "unencoded": unencoded[start:end],
                    "row_uid": uids[start:end],
                    "metadata": {
                        "source_file": np.full(end - start, "capture"),
                        "sequence_index": np.arange(start, end),
                        "device_id": np.full(end - start, "dev"),
                        "raw_attack": behavior[start:end],
                        "behavior_label": behavior[start:end],
                        "timestamp": np.full(end - start, np.nan, dtype=np.float64),
                    },
                    "boolean_raw": {
                        "flag_a": raw_a[start:end],
                        "flag_b": raw_b[start:end],
                    },
                }

        class BatchShapeLinear(torch.nn.Linear):
            def __init__(self, input_features: int) -> None:
                super().__init__(input_features, 2, bias=False)
                self.batch_sizes: list[int] = []

            def forward(self, values):
                self.batch_sizes.append(len(values))
                logits = super().forward(values)
                offset = logits.new_tensor([float(len(values)) * 1e-3, 0.0])
                return logits + offset

        main = BatchShapeLinear(3)
        tiny_model = BatchShapeLinear(1)
        with torch.no_grad():
            main.weight.copy_(torch.tensor([[1., -.5, .25], [-.25, .5, 1.]]))
            tiny_model.weight.copy_(torch.tensor([[1.], [-1.]]))
        main.train()
        tiny_model.train()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with CalibrationCache.create(root / "cache", layout) as cache:
                calibration = populate_validation_cache(
                    cache,
                    batches,
                    main,
                    tiny_model,
                    np.asarray([0]),
                    inference_contract_fingerprint=inference_contract,
                    boolean_fast_path_enabled=True,
                    min_attack_recall=0.75,
                    work_dir=root / "work",
                    encoded_input_dimension=3,
                    chunk_rows=2,
                    stop_after_inference_rows=2,
                )
                self.assertEqual(cache.committed_rows, 2)
                self.assertTrue(main.training)
                self.assertTrue(tiny_model.training)
            with CalibrationCache.open_resume(root / "cache", layout) as cache:
                with self.assertRaisesRegex(ValueError, "Tiny input mismatch"):
                    populate_validation_cache(
                        cache,
                        batches,
                        main,
                        tiny_model,
                        np.asarray([1]),
                        inference_contract_fingerprint=inference_contract,
                        boolean_fast_path_enabled=True,
                        min_attack_recall=0.75,
                        work_dir=root / "work",
                        encoded_input_dimension=3,
                        chunk_rows=2,
                    )
                self.assertEqual(cache.committed_rows, 2)
                with self.assertRaisesRegex(ValueError, "Boolean policy mismatch"):
                    populate_validation_cache(
                        cache,
                        batches,
                        main,
                        tiny_model,
                        np.asarray([0]),
                        inference_contract_fingerprint=inference_contract,
                        boolean_fast_path_enabled=True,
                        min_attack_recall=0.5,
                        work_dir=root / "work",
                        encoded_input_dimension=3,
                        chunk_rows=2,
                    )
                self.assertEqual(cache.committed_rows, 2)
                resumed_calibration = populate_validation_cache(
                    cache,
                    batches,
                    main,
                    tiny_model,
                    np.asarray([0]),
                    inference_contract_fingerprint=inference_contract,
                    boolean_fast_path_enabled=True,
                    min_attack_recall=0.75,
                    work_dir=root / "work",
                    encoded_input_dimension=3,
                    chunk_rows=2,
                )
                self.assertEqual(cache.committed_rows, rows)
                self.assertEqual(resumed_calibration.to_dict(), calibration.to_dict())
                self.assertEqual(main.batch_sizes, [3, 3, 2])
                self.assertEqual(tiny_model.batch_sizes, [3, 3, 2])
                with torch.inference_mode():
                    expected_main = np.concatenate(
                        [
                            torch.softmax(main(torch.from_numpy(features[:3])), dim=1).numpy(),
                            torch.softmax(main(torch.from_numpy(features[3:])), dim=1).numpy(),
                        ]
                    )
                    expected_tiny = np.concatenate(
                        [
                            torch.softmax(
                                tiny_model(torch.from_numpy(features[:3, [0]])), dim=1
                            )[:, 0].numpy(),
                            torch.softmax(
                                tiny_model(torch.from_numpy(features[3:, [0]])), dim=1
                            )[:, 0].numpy(),
                        ]
                    )
                np.testing.assert_allclose(
                    cache.arrays["known_probabilities"], expected_main, rtol=0.0, atol=0.0
                )
                np.testing.assert_allclose(
                    cache.arrays["tiny_benign_probability"],
                    expected_tiny,
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_array_equal(cache.arrays["selected_values"], unencoded)
                expected_flags = np.column_stack(
                    [
                        raw_a <= resumed_calibration.upper_thresholds["flag_a"],
                        raw_b <= resumed_calibration.upper_thresholds["flag_b"],
                    ]
                )
                np.testing.assert_array_equal(cache.arrays["boolean_flags"], expected_flags)
                self.assertEqual(
                    [bytes(value).hex() for value in cache.arrays["uid_digest"]],
                    list(uids),
                )
                exit_calibration = tune_exit_threshold_from_cache(
                    cache, 0.75, 31, 0.05, root / "tiny-route", chunk_rows=2
                )
                route_validation_cache(
                    cache,
                    self._preprocessor(),
                    exit_calibration,
                    self.config,
                    np.asarray([0.0, 1.0, 0.0]),
                    validation_contract=inference_contract,
                    work_dir=root / "route-source-order",
                    chunk_rows=2,
                )
                np.testing.assert_array_equal(
                    cache.arrays["routing_position"], np.arange(rows, dtype=np.int64)
                )

    def test_validation_inference_rejects_contract_before_writing(self) -> None:
        layout = self._layout(2)
        model = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as temporary:
            with CalibrationCache.create(Path(temporary) / "cache", layout) as cache:
                with self.assertRaisesRegex(ValueError, "contract fingerprint mismatch"):
                    populate_validation_cache(
                        cache,
                        lambda: (),
                        model,
                        model,
                        np.asarray([0]),
                        inference_contract_fingerprint="wrong",
                        boolean_fast_path_enabled=False,
                        min_attack_recall=0.5,
                        work_dir=Path(temporary) / "work",
                        encoded_input_dimension=2,
                    )
                self.assertEqual(cache.committed_rows, 0)
                for invalid_indices in (
                    np.asarray([0, 0]),
                    np.asarray([2]),
                    np.asarray([True]),
                ):
                    with self.subTest(indices=invalid_indices.tolist()):
                        with self.assertRaisesRegex(
                            ValueError, "tiny_indices"
                        ):
                            populate_validation_cache(
                                cache,
                                lambda: (),
                                model,
                                model,
                                invalid_indices,
                                inference_contract_fingerprint=(
                                    layout.inference_contract_fingerprint
                                ),
                                boolean_fast_path_enabled=False,
                                min_attack_recall=0.5,
                                work_dir=Path(temporary) / "work",
                                encoded_input_dimension=2,
                            )
                        self.assertEqual(cache.committed_rows, 0)

    def test_calibration_rejects_test_split_before_writing(self) -> None:
        layout = replace(self._layout(2), split="test")
        model = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as temporary:
            with CalibrationCache.create(Path(temporary) / "cache", layout) as cache:
                with self.assertRaisesRegex(ValueError, "validation split"):
                    populate_validation_cache(
                        cache,
                        lambda: (),
                        model,
                        model,
                        np.asarray([0]),
                        inference_contract_fingerprint=layout.inference_contract_fingerprint,
                        boolean_fast_path_enabled=False,
                        min_attack_recall=0.5,
                        work_dir=Path(temporary) / "work",
                        encoded_input_dimension=2,
                    )
                self.assertEqual(cache.committed_rows, 0)

    def test_boolean_thresholds_match_dataframe_across_batches(self) -> None:
        labels = np.asarray(
            ["benign", "scan_like", "benign", "scan_like", "benign", "benign", "scan_like"]
        )
        raw = {
            "a": np.asarray([0.1, 0.2, 0.3, 0.8, 0.4, np.nan, 0.9]),
            "b": np.asarray([0.2, 0.9, np.inf, 0.7, 0.5, 0.1, 0.8]),
        }
        frame = pd.DataFrame({"behavior_label": labels, **raw})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = tune_boolean_fast_path(frame, ["a", "b"], 2.0 / 3.0)
        batches = (
            (labels[:3], {name: value[:3] for name, value in raw.items()}),
            (labels[3:5], {name: value[3:5] for name, value in raw.items()}),
            (labels[5:], {name: value[5:] for name, value in raw.items()}),
        )
        with tempfile.TemporaryDirectory() as temporary:
            actual = tune_boolean_fast_path_streaming(
                batches,
                row_count=len(labels),
                features=("a", "b"),
                min_attack_recall=2.0 / 3.0,
                work_dir=Path(temporary),
                chunk_rows=2,
            )
        self.assertEqual(actual.to_dict(), expected.to_dict())
        np.testing.assert_array_equal(
            apply_boolean_fast_path(frame, actual.upper_thresholds),
            apply_boolean_fast_path(frame, expected.upper_thresholds),
        )

    def test_temporal_routing_cross_batch_resume_matches_in_memory(self) -> None:
        self.config["temporal"]["enabled"] = True
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)
        known = np.asarray(
            [[.9, .1], [.2, .8], [.8, .2], [.3, .7], [.85, .15], [.1, .9]],
            dtype=np.float32,
        )
        selected = np.zeros((6, 2), dtype=np.float32)
        tiny = np.asarray([.95, .70, .90, .65, .88, .55], dtype=np.float32)
        flags = np.zeros((6, 2), dtype=np.bool_)
        timestamps = np.asarray([5., 1., 4., 1., 6., 3.], dtype=np.float64)
        devices = np.asarray(["a", "a", "a", "a", "a", "a"])
        uids = np.asarray([f"{value:064x}" for value in (6, 5, 4, 3, 2, 1)])
        processor = self._preprocessor()
        processor.config["preprocess"]["open_set"]["confidence_threshold"] = 0.0
        calibration = tune_exit_threshold(
            tiny, np.asarray(LABELS)[labels], 0.75, 31, 0.05
        )
        metadata = pd.DataFrame(
            {
                "source_file": "capture",
                "device_id": devices,
                "timestamp": timestamps,
                "sequence_index": np.arange(6),
                "row_uid": uids,
            }
        )
        main_full = np.column_stack((known, np.zeros(6, dtype=np.float32)))
        expected_probabilities, expected_stages, _ = route_with_temporal_state(
            metadata,
            tiny,
            main_full,
            LABELS,
            calibration,
            self.config,
            np.asarray([0.0, 1.0, 0.0]),
            flags.all(axis=1),
        )
        layout = self._layout(6)
        contract = validation_contract(
            {"name": "fixture"},
            cache_base_fingerprint=layout.inference_base_fingerprint,
            config=self.config,
            tiny_indices=np.asarray([0]),
            encoded_input_dimension=2,
        )
        layout = replace(layout, inference_contract_fingerprint=contract)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with CalibrationCache.create(root / "cache", layout) as cache:
                self._commit(
                    cache, labels, known, selected, tiny, flags, timestamps, devices,
                    batch_rows=2,
                    uids=uids,
                )
                with self.assertRaisesRegex(ValueError, "outside the cache layout"):
                    route_validation_cache(
                        cache,
                        processor,
                        calibration,
                        self.config,
                        np.asarray([0.0, 1.0, 0.0]),
                        validation_contract=contract,
                        work_dir=root / "route",
                        chunk_rows=2,
                        stop_after_routed_rows=-1,
                )
                self.assertEqual(cache.routed_committed_rows, 0)
                with self.assertRaisesRegex(
                    ValueError, "contract fingerprint mismatch before routing"
                ):
                    route_validation_cache(
                        cache,
                        processor,
                        calibration,
                        self.config,
                        np.asarray([0.0, 1.0, 0.0]),
                        validation_contract="changed",
                        work_dir=root / "route",
                        chunk_rows=2,
                    )
                self.assertEqual(cache.routed_committed_rows, 0)
                with patch(
                    "bitguard_bnn.out_of_core.calibrate._MAX_MERGE_FAN_IN", 2
                ):
                    fingerprint = route_validation_cache(
                        cache,
                        processor,
                        calibration,
                        self.config,
                        np.asarray([0.0, 1.0, 0.0]),
                        validation_contract=contract,
                        work_dir=root / "route",
                        chunk_rows=2,
                        stop_after_routed_rows=2,
                    )
                self.assertEqual(cache.routed_committed_rows, 2)
                self.assertFalse(list((root / "route").glob("order-*.npy")))
            with CalibrationCache.open_resume(
                root / "cache",
                layout,
                expected_routing_contract_fingerprint=fingerprint,
            ) as cache:
                with self.assertRaisesRegex(
                    ValueError, "contract fingerprint mismatch before routing"
                ):
                    route_validation_cache(
                        cache,
                        processor,
                        calibration,
                        self.config,
                        np.asarray([0.0, 1.0, 0.0]),
                        validation_contract="changed",
                        work_dir=root / "route",
                        chunk_rows=2,
                    )
                self.assertEqual(cache.routed_committed_rows, 2)
                with patch(
                    "bitguard_bnn.out_of_core.calibrate._MAX_MERGE_FAN_IN", 2
                ):
                    route_validation_cache(
                        cache,
                        processor,
                        calibration,
                        self.config,
                        np.asarray([0.0, 1.0, 0.0]),
                        validation_contract=contract,
                        work_dir=root / "route",
                        chunk_rows=2,
                    )
                np.testing.assert_allclose(
                    cache.arrays["routed_probabilities"], expected_probabilities,
                    rtol=0.0, atol=1e-7,
                )
                np.testing.assert_array_equal(cache.arrays["exit_stage"], expected_stages)

    def test_fixed_fpr_tied_float32_matches_existing_float64_semantics(self) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int32)
        known = np.asarray(
            [[.9, .1], [.8, .2], [.8, .2], [.7, .3], [.2, .8], [.1, .9]],
            dtype=np.float32,
        )
        selected = np.zeros((6, 2), dtype=np.float32)
        tiny = known[:, 0].copy()
        flags = np.zeros((6, 2), dtype=np.bool_)
        routed = np.column_stack((known, np.zeros(6, dtype=np.float32)))
        expected = calibrate_fixed_fpr_thresholds(
            np.asarray(LABELS)[labels], LABELS, routed, (0.25, 0.5)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with CalibrationCache.create(root / "cache", self._layout(6)) as cache:
                self._commit(
                    cache, labels, known, selected, tiny, flags,
                    np.arange(6, dtype=np.float64), np.full(6, "d", dtype="<U1"),
                )
                cache.commit_routing_range(
                    0,
                    {
                        "cache_position": np.arange(6, dtype=np.int64),
                        "routed_probabilities": routed.astype(np.float32),
                        "exit_stage": np.full(6, 2, dtype=np.int16),
                    },
                    routing_contract_fingerprint="route",
                )
                actual = calibrate_fixed_fpr_from_cache(
                    cache, (0.25, 0.5), root / "fpr", chunk_rows=2
                )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
