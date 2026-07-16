from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bitguard_bnn.cli import main
from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.metrics import classification_metrics
import bitguard_bnn.out_of_core.evaluate as evaluation_module
import bitguard_bnn.out_of_core.replay as replay_module
from bitguard_bnn.out_of_core.evaluate import evaluate_prediction_batches
from bitguard_bnn.out_of_core.metrics import StreamingClassificationMetrics
from bitguard_bnn.out_of_core.replay import replay_parquet_predictions
from bitguard_bnn.state import (
    TemporalSecurityStateMachine,
    replay_prediction_row,
    replay_predictions,
    temporal_state_key,
)


class _StringKey:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class StreamingMetricParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = ["benign", "scan_like", "flood_like", "unknown_like"]
        self.y_true = np.asarray(
            [
                "benign",
                "scan_like",
                "benign",
                "flood_like",
                "unknown_like",
                "scan_like",
                "benign",
                "flood_like",
                "unknown_like",
            ]
        )
        # Includes score ties and probability predictions that differ from the
        # deployed label, exercising both calibration and confusion semantics.
        self.probabilities = np.asarray(
            [
                [0.8, 0.1, 0.05, 0.05],
                [0.2, 0.6, 0.1, 0.1],
                [0.5, 0.2, 0.2, 0.1],
                [0.2, 0.2, 0.5, 0.1],
                [0.2, 0.2, 0.1, 0.5],
                [0.2, 0.6, 0.1, 0.1],
                [0.5, 0.2, 0.2, 0.1],
                [0.6, 0.1, 0.2, 0.1],
                [0.1, 0.2, 0.2, 0.5],
            ],
            dtype=np.float64,
        )
        self.y_pred = np.asarray(
            ["benign", "scan_like", "benign", "flood_like", "unknown_like", "scan_like", "benign", "benign", "unknown_like"]
        )
        self.thresholds = {0.5: 0.5, 0.25: 0.61}

    def _expected(self) -> dict[str, object]:
        return classification_metrics(
            self.y_true,
            self.y_pred,
            self.labels,
            self.probabilities,
            ["scan_like", "flood_like", "unknown_like"],
            self.thresholds,
        )

    def _assert_nested_close(self, expected: object, actual: object) -> None:
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict)
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self._assert_nested_close(expected[key], actual[key])
        elif isinstance(expected, float):
            self.assertAlmostEqual(expected, actual, delta=1e-12)
        else:
            self.assertEqual(expected, actual)

    def test_matches_in_memory_at_every_batch_boundary(self) -> None:
        expected = self._expected()
        for boundary in range(1, len(self.y_true)):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                accumulator = StreamingClassificationMetrics(
                    probability_labels=self.labels,
                    high_risk_labels=["scan_like", "flood_like", "unknown_like"],
                    temporary_directory=Path(tmp),
                    score_run_rows=2,
                )
                for start, end in ((0, boundary), (boundary, len(self.y_true))):
                    accumulator.update(
                        self.y_true[start:end],
                        self.y_pred[start:end],
                        self.probabilities[start:end],
                        np.asarray([f"uid-{index:03d}" for index in range(start, end)]),
                    )
                actual = accumulator.finalize(self.thresholds)
                self._assert_nested_close(expected, actual)
                accumulator.cleanup()

    def test_absent_probability_class_is_omitted_from_ordered_metrics(self) -> None:
        labels = [*self.labels, "beacon_like"]
        probabilities = np.column_stack([self.probabilities, np.zeros(len(self.y_true))])
        with tempfile.TemporaryDirectory() as tmp:
            accumulator = StreamingClassificationMetrics(
                probability_labels=labels,
                high_risk_labels=[],
                temporary_directory=Path(tmp),
                score_run_rows=3,
            )
            accumulator.update(
                self.y_true,
                self.y_pred,
                probabilities,
                np.asarray([f"uid-{index:03d}" for index in range(len(self.y_true))]),
            )
            actual = accumulator.finalize()
            self.assertNotIn("beacon_like", actual["auroc_per_class"])
            self.assertNotIn("beacon_like", actual["auprc_per_class"])
            accumulator.cleanup()

    def test_three_way_merge_and_ties_across_run_boundaries(self) -> None:
        expected = self._expected()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accumulators = [
                StreamingClassificationMetrics(
                    probability_labels=self.labels,
                    high_risk_labels=["scan_like", "flood_like", "unknown_like"],
                    temporary_directory=root,
                    score_run_rows=1,
                )
                for _ in range(3)
            ]
            for accumulator, indexes in zip(
                accumulators,
                (np.asarray([0, 3, 6]), np.asarray([1, 4, 7]), np.asarray([2, 5, 8])),
                strict=True,
            ):
                accumulator.update(
                    self.y_true[indexes],
                    self.y_pred[indexes],
                    self.probabilities[indexes],
                    np.asarray([f"uid-{index:03d}" for index in indexes]),
                )
            accumulators[0].merge(accumulators[1])
            accumulators[0].merge(accumulators[2])
            with patch("bitguard_bnn.out_of_core.metrics._MAX_MERGE_FAN_IN", 2):
                self._assert_nested_close(
                    expected, accumulators[0].finalize(self.thresholds)
                )
            run_roots = [accumulator.root for accumulator in accumulators]
            for accumulator in accumulators:
                accumulator.cleanup()
            self.assertTrue(all(not path.exists() for path in run_roots))

    def test_real_fan_in_compaction_survives_finalize_update_and_merge(self) -> None:
        templates = np.asarray(
            [
                [0.5, 0.3, 0.1, 0.1],
                [0.2, 0.5, 0.2, 0.1],
                [0.2, 0.2, 0.5, 0.1],
                [0.2, 0.1, 0.2, 0.5],
            ],
            dtype=np.float64,
        )

        def fixture(start: int, rows: int):
            indexes = np.arange(start, start + rows)
            true = np.asarray([self.labels[index % 4] for index in indexes])
            probabilities = templates[indexes % len(templates)]
            predicted = np.asarray(
                [self.labels[index] for index in probabilities.argmax(axis=1)]
            )
            uids = np.asarray([f"compaction-{index:05d}" for index in indexes])
            return true, predicted, probabilities, uids

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = StreamingClassificationMetrics(
                probability_labels=self.labels,
                high_risk_labels=["scan_like", "flood_like", "unknown_like"],
                temporary_directory=root,
                score_run_rows=1,
            )
            first = fixture(0, 70)
            primary.update(*first)
            first_expected = classification_metrics(
                first[0], first[1], self.labels, first[2],
                ["scan_like", "flood_like", "unknown_like"],
            )
            self._assert_nested_close(first_expected, primary.finalize())

            second = fixture(70, 70)
            primary.update(*second)
            secondary = StreamingClassificationMetrics(
                probability_labels=self.labels,
                high_risk_labels=["scan_like", "flood_like", "unknown_like"],
                temporary_directory=root,
                score_run_rows=1,
            )
            third = fixture(140, 40)
            secondary.update(*third)
            primary.merge(secondary)
            combined = tuple(
                np.concatenate([first[index], second[index], third[index]])
                for index in range(4)
            )
            expected = classification_metrics(
                combined[0], combined[1], self.labels, combined[2],
                ["scan_like", "flood_like", "unknown_like"],
            )
            self._assert_nested_close(expected, primary.finalize())
            roots = (primary.root, secondary.root)
            primary.cleanup()
            secondary.cleanup()
            self.assertTrue(all(not path.exists() for path in roots))


class StreamingPredictionArtifactTests(unittest.TestCase):
    labels = ["benign", "scan_like", "unknown_like"]

    def _assert_nested_close(self, expected: object, actual: object) -> None:
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict)
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self._assert_nested_close(expected[key], actual[key])
        elif isinstance(expected, float):
            self.assertAlmostEqual(expected, actual, delta=1e-12)
        else:
            self.assertEqual(expected, actual)

    @staticmethod
    def _rows() -> dict[str, np.ndarray]:
        return {
            "row_uid": np.asarray(["uid-c", "uid-a", "uid-e", "uid-b", "uid-d"]),
            "true_label": np.asarray(["benign", "scan_like", "unknown_like", "benign", "scan_like"]),
            "predicted_label": np.asarray(["benign", "scan_like", "unknown_like", "scan_like", "scan_like"]),
            "probabilities": np.asarray(
                [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.4, 0.5, 0.1], [0.2, 0.7, 0.1]],
                dtype=np.float32,
            ),
            "exit_stage": np.asarray([0, 2, 2, 1, 2], dtype=np.int16),
            "source_file": np.asarray(["episode"] * 5),
            "device_id": np.asarray(["device-a", "device-a", "device-b", "device-a", "device-b"]),
            "timestamp": np.asarray([3.0, 1.0, 5.0, 2.0, 4.0]),
            "sequence_index": np.asarray([3, 1, 5, 2, 4], dtype=np.int64),
            "raw_attack": np.asarray(["benign", "scan", "unknown", "benign", "scan"]),
            "has_wall_clock_time": np.asarray([True] * 5),
            "temporal_continuity": np.asarray([True] * 5),
        }

    @classmethod
    def _batches(cls, boundaries: tuple[int, ...]) -> list[dict[str, np.ndarray]]:
        rows = cls._rows()
        points = (0, *boundaries, len(rows["row_uid"]))
        return [
            {key: value[start:end] for key, value in rows.items()}
            for start, end in zip(points[:-1], points[1:], strict=True)
        ]

    def _evaluate(
        self,
        root: Path,
        boundaries: tuple[int, ...],
        *,
        temporary_directory: Path | None = None,
    ) -> dict[str, object]:
        return evaluate_prediction_batches(
            lambda: iter(self._batches(boundaries)),
            probability_labels=self.labels,
            high_risk_labels=["scan_like", "unknown_like"],
            test_contract="fixture-test-contract-v1",
            operating_thresholds={0.5: 0.5},
            prediction_path=root / "predictions.parquet",
            metrics_path=root / "metrics.json",
            plot_sample_path=root / "plot_sample.parquet",
            plot_manifest_path=root / "plot_manifest.json",
            temporary_directory=(
                root / "temporary"
                if temporary_directory is None
                else temporary_directory
            ),
            plot_sample_rows=3,
            plot_sample_seed=2309,
            score_run_rows=2,
        )

    def _evaluate_rows(
        self,
        root: Path,
        rows: dict[str, np.ndarray],
        *,
        test_contract: str,
    ) -> dict[str, object]:
        return evaluate_prediction_batches(
            lambda: iter([rows]),
            probability_labels=self.labels,
            high_risk_labels=["scan_like", "unknown_like"],
            test_contract=test_contract,
            prediction_path=root / "predictions.parquet",
            metrics_path=root / "metrics.json",
            plot_sample_path=root / "plot_sample.parquet",
            plot_manifest_path=root / "plot_manifest.json",
            temporary_directory=root / "evaluation-temporary",
            plot_sample_rows=3,
            plot_sample_seed=2309,
            score_run_rows=2,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def test_atomic_compressed_parquet_and_sample_ignore_batch_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            first_result = self._evaluate(first, (1, 3))
            second_result = self._evaluate(second, (2, 4))
            self.assertEqual(first_result["metrics"], second_result["metrics"])
            metadata = pq.ParquetFile(first / "predictions.parquet").metadata
            self.assertEqual(metadata.num_rows, 5)
            self.assertEqual(metadata.row_group(0).column(0).compression, "ZSTD")
            arrow_metadata = pq.ParquetFile(first / "predictions.parquet").schema_arrow.metadata
            self.assertEqual(arrow_metadata[b"bitguard_artifact_format"], b"prediction_parquet")
            self.assertEqual(arrow_metadata[b"bitguard_artifact_version"], b"1")
            self.assertEqual(
                arrow_metadata[b"bitguard_probability_labels"],
                b'["benign","scan_like","unknown_like"]',
            )
            persisted = pq.read_table(first / "predictions.parquet").to_pandas()
            persisted_probabilities = persisted[
                [f"prob_{label}" for label in self.labels]
            ].to_numpy()
            expected_metrics = classification_metrics(
                persisted["true_label"].to_numpy(),
                persisted["predicted_label"].to_numpy(),
                self.labels,
                persisted_probabilities,
                ["scan_like", "unknown_like"],
                {0.5: 0.5},
            )
            self._assert_nested_close(expected_metrics, first_result["metrics"])
            sample_one = pq.read_table(first / "plot_sample.parquet").to_pandas()
            sample_two = pq.read_table(second / "plot_sample.parquet").to_pandas()
            self.assertEqual(sample_one["row_uid"].tolist(), sample_two["row_uid"].tolist())
            self.assertEqual(
                first_result["plot_manifest"]["numeric_metrics_scope"], "full_test"
            )
            self.assertEqual(
                first_result["plot_manifest"]["plot_rows_scope"], "deterministic_sample"
            )
            self.assertFalse((first / "temporary").exists())

    def test_batch_failure_publishes_nothing_and_releases_temporaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def failing_batches():
                yield self._batches((2,))[0]
                raise RuntimeError("injected prediction failure")

            with self.assertRaisesRegex(RuntimeError, "injected prediction failure"):
                evaluate_prediction_batches(
                    failing_batches,
                    probability_labels=self.labels,
                    high_risk_labels=[],
                    test_contract="fixture-test-contract-v1",
                    prediction_path=root / "predictions.parquet",
                    metrics_path=root / "metrics.json",
                    plot_sample_path=root / "sample.parquet",
                    plot_manifest_path=root / "sample.json",
                    temporary_directory=root / "temporary",
                    plot_sample_rows=2,
                )
            self.assertFalse((root / "predictions.parquet").exists())
            self.assertFalse((root / "metrics.json").exists())
            self.assertFalse((root / "temporary").exists())
            self.assertEqual(list(root.glob("*.partial*")), [])

    def test_global_duplicate_uid_is_rejected_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batches = self._batches((2,))
            batches[1]["row_uid"][0] = batches[0]["row_uid"][0]
            with self.assertRaisesRegex(ValueError, "globally unique"):
                evaluate_prediction_batches(
                    lambda: iter(batches),
                    probability_labels=self.labels,
                    high_risk_labels=[],
                    test_contract="fixture-test-contract-v1",
                    prediction_path=root / "predictions.parquet",
                    metrics_path=root / "metrics.json",
                    plot_sample_path=root / "sample.parquet",
                    plot_manifest_path=root / "sample.json",
                    temporary_directory=root / "temporary",
                    plot_sample_rows=2,
                )
            self.assertEqual(list(root.glob("*.parquet")), [])
            self.assertFalse((root / "temporary").exists())

    def test_reserved_and_string_normalized_fields_are_rejected_before_writes(self) -> None:
        collisions: list[object] = [
            "storage_position",
            "prob_benign",
            _StringKey("storage_position"),
            _StringKey("prob_benign"),
        ]
        for collision in collisions:
            with self.subTest(collision=str(collision)), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                batch = self._batches((2,))[0]
                batch[collision] = np.arange(len(batch["row_uid"]))
                with (
                    patch.object(
                        evaluation_module.sqlite3,
                        "connect",
                        side_effect=AssertionError("UID database opened before validation"),
                    ) as connect,
                    patch.object(
                        evaluation_module.pq,
                        "ParquetWriter",
                        side_effect=AssertionError("Parquet opened before validation"),
                    ) as writer,
                    self.assertRaisesRegex(ValueError, "reserved|collision"),
                ):
                    evaluate_prediction_batches(
                        lambda: iter([batch]),
                        probability_labels=self.labels,
                        high_risk_labels=[],
                        test_contract="fixture-test-contract-v1",
                        prediction_path=root / "predictions.parquet",
                        metrics_path=root / "metrics.json",
                        plot_sample_path=root / "sample.parquet",
                        plot_manifest_path=root / "sample.json",
                        temporary_directory=root / "temporary",
                        plot_sample_rows=2,
                    )
                connect.assert_not_called()
                writer.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])

    def test_exit_stage_contract_is_required_and_validated_before_writes(self) -> None:
        cases: list[tuple[str, object]] = [
            ("missing", None),
            ("noninteger", np.asarray([0.0, 1.0])),
            ("negative", np.asarray([0, -1], dtype=np.int16)),
            ("too_large", np.asarray([0, 3], dtype=np.int16)),
        ]
        for name, replacement in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                batch = self._batches((2,))[0]
                if replacement is None:
                    batch.pop("exit_stage")
                else:
                    batch["exit_stage"] = replacement
                with (
                    patch.object(
                        evaluation_module.sqlite3,
                        "connect",
                        side_effect=AssertionError("UID database opened before validation"),
                    ) as connect,
                    patch.object(
                        evaluation_module.pq,
                        "ParquetWriter",
                        side_effect=AssertionError("Parquet opened before validation"),
                    ) as writer,
                    self.assertRaisesRegex(ValueError, "exit_stage|missing"),
                ):
                    evaluate_prediction_batches(
                        lambda: iter([batch]),
                        probability_labels=self.labels,
                        high_risk_labels=[],
                        test_contract="fixture-test-contract-v1",
                        prediction_path=root / "predictions.parquet",
                        metrics_path=root / "metrics.json",
                        plot_sample_path=root / "sample.parquet",
                        plot_manifest_path=root / "sample.json",
                        temporary_directory=root / "temporary",
                        plot_sample_rows=2,
                    )
                connect.assert_not_called()
                writer.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])

    def test_publish_transaction_recovers_every_interrupted_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as baseline_tmp:
            baseline_root = Path(baseline_tmp)
            baseline = self._evaluate(baseline_root, (1, 3))
            baseline_tables = {
                name: pq.read_table(baseline_root / name).to_pydict()
                for name in ("predictions.parquet", "plot_sample.parquet")
            }
            baseline_json = {
                name: json.loads((baseline_root / name).read_text(encoding="utf-8"))
                for name in ("metrics.json", "plot_manifest.json")
            }
            self.assertEqual(baseline["metrics"], baseline_json["metrics.json"])

            for boundary in range(4):
                with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)

                    def interrupt(index: int, _path: Path) -> None:
                        if index == boundary:
                            raise RuntimeError(f"simulated hard interruption {boundary}")

                    with (
                        patch.object(
                            evaluation_module,
                            "_after_publish_boundary",
                            side_effect=interrupt,
                            create=True,
                        ),
                        self.assertRaisesRegex(RuntimeError, "simulated hard interruption"),
                    ):
                        self._evaluate(root, (1, 3))
                    foreign = root / "foreign.txt"
                    foreign.write_text("do-not-touch", encoding="utf-8")

                    def must_not_reinfer():
                        raise AssertionError("recovery must not rerun inference")

                    recovered = evaluate_prediction_batches(
                        must_not_reinfer,
                        probability_labels=self.labels,
                        high_risk_labels=["scan_like", "unknown_like"],
                        test_contract="fixture-test-contract-v1",
                        operating_thresholds={0.5: 0.5},
                        prediction_path=root / "predictions.parquet",
                        metrics_path=root / "metrics.json",
                        plot_sample_path=root / "plot_sample.parquet",
                        plot_manifest_path=root / "plot_manifest.json",
                        temporary_directory=root / "temporary",
                        plot_sample_rows=3,
                        plot_sample_seed=2309,
                        score_run_rows=2,
                    )
                    expected_result = dict(baseline)
                    expected_result["prediction_path"] = str(
                        (root / "predictions.parquet").resolve()
                    )
                    self.assertEqual(recovered, expected_result)
                    reused = evaluate_prediction_batches(
                        must_not_reinfer,
                        probability_labels=self.labels,
                        high_risk_labels=["scan_like", "unknown_like"],
                        test_contract="fixture-test-contract-v1",
                        operating_thresholds={0.5: 0.5},
                        prediction_path=root / "predictions.parquet",
                        metrics_path=root / "metrics.json",
                        plot_sample_path=root / "plot_sample.parquet",
                        plot_manifest_path=root / "plot_manifest.json",
                        temporary_directory=root / "temporary",
                        plot_sample_rows=3,
                        plot_sample_seed=2309,
                        score_run_rows=2,
                    )
                    self.assertEqual(reused, expected_result)
                    for name, expected in baseline_tables.items():
                        self.assertEqual(pq.read_table(root / name).to_pydict(), expected)
                    for name, expected in baseline_json.items():
                        self.assertEqual(
                            json.loads((root / name).read_text(encoding="utf-8")),
                            expected,
                        )
                    self.assertEqual(foreign.read_text(encoding="utf-8"), "do-not-touch")
                    markers = [
                        path
                        for path in root.iterdir()
                        if path.name.endswith(".evaluation-transaction.json")
                    ]
                    self.assertEqual(len(markers), 1)
                    self.assertEqual(
                        json.loads(markers[0].read_text(encoding="utf-8"))["state"],
                        "committed",
                    )

    def test_publish_recovery_never_deletes_a_foreign_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def interrupt(index: int, _path: Path) -> None:
                if index == 0:
                    raise RuntimeError("simulated hard interruption")

            with (
                patch.object(
                    evaluation_module,
                    "_after_publish_boundary",
                    side_effect=interrupt,
                    create=True,
                ),
                self.assertRaisesRegex(RuntimeError, "simulated hard interruption"),
            ):
                self._evaluate(root, (1, 3))
            prediction = root / "predictions.parquet"
            prediction.write_bytes(b"foreign replacement")
            with self.assertRaisesRegex(RuntimeError, "foreign|identity"):
                self._evaluate(root, (1, 3))
            self.assertEqual(prediction.read_bytes(), b"foreign replacement")

    def test_evaluation_atomic_publish_never_clobbers_or_commits_foreign_entries(self) -> None:
        for mode in ("before_link", "before_commit"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prediction = root / "predictions.parquet"
                foreign_instance: list[tuple[int, int]] = []

                def foreign_before(index: int, final: Path) -> None:
                    if index == 0:
                        final.write_bytes(b"foreign-before-link")

                def foreign_after(index: int, final: Path) -> None:
                    if index == 3:
                        same_bytes = prediction.read_bytes()
                        prediction.unlink()
                        prediction.write_bytes(same_bytes)
                        observed = prediction.stat(follow_symlinks=False)
                        foreign_instance.append(
                            (int(observed.st_dev), int(observed.st_ino))
                        )

                hook = (
                    patch.object(
                        evaluation_module,
                        "_before_atomic_publish",
                        side_effect=foreign_before,
                    )
                    if mode == "before_link"
                    else patch.object(
                        evaluation_module,
                        "_after_publish_boundary",
                        side_effect=foreign_after,
                    )
                )
                with hook, self.assertRaisesRegex(RuntimeError, "foreign|identity"):
                    self._evaluate(root, (1, 3))
                if mode == "before_link":
                    self.assertEqual(prediction.read_bytes(), b"foreign-before-link")
                else:
                    observed = prediction.stat(follow_symlinks=False)
                    self.assertEqual(
                        (int(observed.st_dev), int(observed.st_ino)),
                        foreign_instance[0],
                    )

    def test_evaluation_rejects_dangling_final_and_partial_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dangling_final = root / "predictions.parquet"
            try:
                dangling_final.symlink_to(root / "missing-final-target")
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaises(FileExistsError):
                self._evaluate(root, (1, 3))
            self.assertTrue(os.path.lexists(dangling_final))
            self.assertTrue(dangling_final.is_symlink())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics.json"
            prediction = root / "predictions.parquet"
            sample = root / "plot_sample.parquet"
            plot_manifest = root / "plot_manifest.json"
            publish_order = [prediction, sample, plot_manifest, metrics]
            transaction_id = "a" * 32
            temporary_parent = (root / "temporary").resolve()
            work = temporary_parent / f"evaluation-{transaction_id}"
            work.mkdir(parents=True)
            owner_payload = {
                "format_version": 1,
                "transaction_id": transaction_id,
                "test_contract": "fixture-test-contract-v1",
            }
            owner_marker = work / evaluation_module._WORK_OWNER_FILE
            evaluation_module._write_json(owner_marker, owner_payload)
            partials = {
                final: final.with_name(f".{final.name}.{transaction_id}.partial")
                for final in publish_order
            }
            dangling_partial = partials[prediction]
            dangling_partial.symlink_to(root / "missing-partial-target")
            request_contract = {
                "probability_labels": self.labels,
                "high_risk_labels": ["scan_like", "unknown_like"],
                "operating_thresholds": [[0.5, 0.5]],
                "plot_sample_rows": 3,
                "plot_sample_seed": "2309",
            }
            transaction = {
                "format_version": evaluation_module._EVALUATION_TRANSACTION_VERSION,
                "state": "building",
                "transaction_id": transaction_id,
                "test_contract": "fixture-test-contract-v1",
                "request_contract": request_contract,
                "owned_work": {
                    "path": str(work),
                    "owner_marker": str(owner_marker),
                    "owner_initializing_marker": str(
                        work / evaluation_module._WORK_OWNER_INITIALIZING_FILE
                    ),
                    "owner_payload": owner_payload,
                    "temporary_parent": str(temporary_parent),
                    "parent_created": False,
                },
                "artifacts": [
                    {
                        "final": str(final),
                        "partial": str(partials[final]),
                        "identity": None,
                        "instance": None,
                    }
                    for final in publish_order
                ],
            }
            evaluation_module._write_transaction(
                evaluation_module._transaction_path(metrics), transaction
            )
            with self.assertRaisesRegex(RuntimeError, "foreign partial"):
                self._evaluate(root, (1, 3))
            self.assertTrue(os.path.lexists(dangling_partial))
            self.assertTrue(dangling_partial.is_symlink())

    def test_process_exit_publish_recovery_removes_only_owned_work(self) -> None:
        with tempfile.TemporaryDirectory() as baseline_tmp, tempfile.TemporaryDirectory() as tmp:
            baseline_root = Path(baseline_tmp)
            baseline = self._evaluate(baseline_root, (1, 3))
            root = Path(tmp)
            script = textwrap.dedent(
                f"""
                import os
                from pathlib import Path
                import numpy as np
                import bitguard_bnn.out_of_core.evaluate as module

                root = Path({str(root)!r})
                rows = {{
                    "row_uid": np.asarray(["uid-c", "uid-a", "uid-e", "uid-b", "uid-d"]),
                    "true_label": np.asarray(["benign", "scan_like", "unknown_like", "benign", "scan_like"]),
                    "predicted_label": np.asarray(["benign", "scan_like", "unknown_like", "scan_like", "scan_like"]),
                    "probabilities": np.asarray([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.2,0.7],[0.4,0.5,0.1],[0.2,0.7,0.1]], dtype=np.float32),
                    "exit_stage": np.asarray([0,2,2,1,2], dtype=np.int16),
                    "source_file": np.asarray(["episode"] * 5),
                    "device_id": np.asarray(["device-a","device-a","device-b","device-a","device-b"]),
                    "timestamp": np.asarray([3.0,1.0,5.0,2.0,4.0]),
                    "sequence_index": np.asarray([3,1,5,2,4], dtype=np.int64),
                    "raw_attack": np.asarray(["benign","scan","unknown","benign","scan"]),
                    "has_wall_clock_time": np.asarray([True] * 5),
                    "temporal_continuity": np.asarray([True] * 5),
                }}
                def batches():
                    for start, end in ((0,1),(1,3),(3,5)):
                        yield {{key: value[start:end] for key, value in rows.items()}}
                def hard_exit(index, path):
                    if index == 1:
                        os._exit(91)
                module._after_publish_link_boundary = hard_exit
                module.evaluate_prediction_batches(
                    batches,
                    probability_labels=["benign","scan_like","unknown_like"],
                    high_risk_labels=["scan_like","unknown_like"],
                    test_contract="fixture-test-contract-v1",
                    operating_thresholds={{0.5: 0.5}},
                    prediction_path=root / "predictions.parquet",
                    metrics_path=root / "metrics.json",
                    plot_sample_path=root / "plot_sample.parquet",
                    plot_manifest_path=root / "plot_manifest.json",
                    temporary_directory=root / "temporary",
                    plot_sample_rows=3,
                    plot_sample_seed=2309,
                    score_run_rows=2,
                )
                """
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(
                (Path(__file__).resolve().parents[1] / "src").resolve()
            )
            child = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(child.returncode, 91, child.stderr)
            work_parent = root / "temporary"
            owned = list(work_parent.glob("evaluation-*"))
            self.assertEqual(len(owned), 1)
            self.assertTrue(any(owned[0].rglob("row-uids.sqlite")))
            foreign = work_parent / "foreign-sibling.txt"
            foreign.write_text("preserve", encoding="utf-8")

            def must_not_reinfer():
                raise AssertionError("publication recovery must not rerun inference")

            recovered = evaluate_prediction_batches(
                must_not_reinfer,
                probability_labels=self.labels,
                high_risk_labels=["scan_like", "unknown_like"],
                test_contract="fixture-test-contract-v1",
                operating_thresholds={0.5: 0.5},
                prediction_path=root / "predictions.parquet",
                metrics_path=root / "metrics.json",
                plot_sample_path=root / "plot_sample.parquet",
                plot_manifest_path=root / "plot_manifest.json",
                temporary_directory=work_parent,
                plot_sample_rows=3,
                plot_sample_seed=2309,
                score_run_rows=2,
            )
            expected = dict(baseline)
            expected["prediction_path"] = str((root / "predictions.parquet").resolve())
            self.assertEqual(recovered, expected)
            self.assertEqual(list(work_parent.glob("evaluation-*")), [])
            self.assertEqual(foreign.read_text(encoding="utf-8"), "preserve")

    def test_process_exit_during_build_cleans_owned_work_and_recomputes(self) -> None:
        with tempfile.TemporaryDirectory() as baseline_tmp, tempfile.TemporaryDirectory() as tmp:
            baseline_root = Path(baseline_tmp)
            baseline = self._evaluate(baseline_root, (1, 3))
            root = Path(tmp)
            script = textwrap.dedent(
                f"""
                import os
                from pathlib import Path
                import numpy as np
                import bitguard_bnn.out_of_core.evaluate as module

                root = Path({str(root)!r})
                rows = {{
                    "row_uid": np.asarray(["uid-c", "uid-a", "uid-e", "uid-b", "uid-d"]),
                    "true_label": np.asarray(["benign", "scan_like", "unknown_like", "benign", "scan_like"]),
                    "predicted_label": np.asarray(["benign", "scan_like", "unknown_like", "scan_like", "scan_like"]),
                    "probabilities": np.asarray([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.2,0.7],[0.4,0.5,0.1],[0.2,0.7,0.1]], dtype=np.float32),
                    "exit_stage": np.asarray([0,2,2,1,2], dtype=np.int16),
                    "source_file": np.asarray(["episode"] * 5),
                    "device_id": np.asarray(["device-a","device-a","device-b","device-a","device-b"]),
                    "timestamp": np.asarray([3.0,1.0,5.0,2.0,4.0]),
                    "sequence_index": np.asarray([3,1,5,2,4], dtype=np.int64),
                    "raw_attack": np.asarray(["benign","scan","unknown","benign","scan"]),
                    "has_wall_clock_time": np.asarray([True] * 5),
                    "temporal_continuity": np.asarray([True] * 5),
                }}
                def batches():
                    for start, end in ((0,1),(1,3),(3,5)):
                        yield {{key: value[start:end] for key, value in rows.items()}}
                def hard_exit(index, path):
                    if index == 0:
                        os._exit(92)
                module._after_build_boundary = hard_exit
                module.evaluate_prediction_batches(
                    batches,
                    probability_labels=["benign","scan_like","unknown_like"],
                    high_risk_labels=["scan_like","unknown_like"],
                    test_contract="fixture-test-contract-v1",
                    operating_thresholds={{0.5: 0.5}},
                    prediction_path=root / "predictions.parquet",
                    metrics_path=root / "metrics.json",
                    plot_sample_path=root / "plot_sample.parquet",
                    plot_manifest_path=root / "plot_manifest.json",
                    temporary_directory=root / "temporary",
                    plot_sample_rows=3,
                    plot_sample_seed=2309,
                    score_run_rows=2,
                )
                """
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(
                (Path(__file__).resolve().parents[1] / "src").resolve()
            )
            child = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(child.returncode, 92, child.stderr)
            journal = root / ".metrics.json.evaluation-transaction.json"
            self.assertEqual(
                json.loads(journal.read_text(encoding="utf-8"))["state"],
                "building",
            )
            self.assertFalse(any((root / name).exists() for name in (
                "predictions.parquet",
                "metrics.json",
                "plot_sample.parquet",
                "plot_manifest.json",
            )))
            self.assertEqual(len(list(root.glob(".*.partial"))), 1)
            work_parent = root / "temporary"
            self.assertEqual(len(list(work_parent.glob("evaluation-*"))), 1)
            foreign = work_parent / "foreign-sibling.txt"
            foreign.write_text("preserve", encoding="utf-8")
            recomputations = 0

            def recompute():
                nonlocal recomputations
                recomputations += 1
                return iter(self._batches((1, 3)))

            recovered = evaluate_prediction_batches(
                recompute,
                probability_labels=self.labels,
                high_risk_labels=["scan_like", "unknown_like"],
                test_contract="fixture-test-contract-v1",
                operating_thresholds={0.5: 0.5},
                prediction_path=root / "predictions.parquet",
                metrics_path=root / "metrics.json",
                plot_sample_path=root / "plot_sample.parquet",
                plot_manifest_path=root / "plot_manifest.json",
                temporary_directory=work_parent,
                plot_sample_rows=3,
                plot_sample_seed=2309,
                score_run_rows=2,
            )
            expected = dict(baseline)
            expected["prediction_path"] = str((root / "predictions.parquet").resolve())
            self.assertEqual(recovered, expected)
            self.assertEqual(recomputations, 1)
            self.assertEqual(list(work_parent.glob("evaluation-*")), [])
            self.assertEqual(foreign.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(list(root.glob(".*.partial")), [])

    def test_process_exit_during_work_ownership_initialization_is_recoverable(self) -> None:
        for mode, exit_code in (("after_mkdir", 93), ("during_marker_write", 94)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                hook = (
                    "module._after_work_directory_created = "
                    f"lambda path: os._exit({exit_code})"
                    if mode == "after_mkdir"
                    else "module._during_owner_marker_write = "
                    f"lambda path: os._exit({exit_code})"
                )
                script = textwrap.dedent(
                    f"""
                    import os
                    from pathlib import Path
                    import numpy as np
                    import bitguard_bnn.out_of_core.evaluate as module

                    root = Path({str(root)!r})
                    rows = {{
                        "row_uid": np.asarray(["uid-c", "uid-a", "uid-e", "uid-b", "uid-d"]),
                        "true_label": np.asarray(["benign", "scan_like", "unknown_like", "benign", "scan_like"]),
                        "predicted_label": np.asarray(["benign", "scan_like", "unknown_like", "scan_like", "scan_like"]),
                        "probabilities": np.asarray([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.2,0.7],[0.4,0.5,0.1],[0.2,0.7,0.1]], dtype=np.float32),
                        "exit_stage": np.asarray([0,2,2,1,2], dtype=np.int16),
                        "source_file": np.asarray(["episode"] * 5),
                        "device_id": np.asarray(["device-a","device-a","device-b","device-a","device-b"]),
                        "timestamp": np.asarray([3.0,1.0,5.0,2.0,4.0]),
                        "sequence_index": np.asarray([3,1,5,2,4], dtype=np.int64),
                        "raw_attack": np.asarray(["benign","scan","unknown","benign","scan"]),
                        "has_wall_clock_time": np.asarray([True] * 5),
                        "temporal_continuity": np.asarray([True] * 5),
                    }}
                    def batches():
                        yield rows
                    {hook}
                    module.evaluate_prediction_batches(
                        batches,
                        probability_labels=["benign","scan_like","unknown_like"],
                        high_risk_labels=["scan_like","unknown_like"],
                        test_contract="fixture-test-contract-v1",
                        operating_thresholds={{0.5: 0.5}},
                        prediction_path=root / "predictions.parquet",
                        metrics_path=root / "metrics.json",
                        plot_sample_path=root / "plot_sample.parquet",
                        plot_manifest_path=root / "plot_manifest.json",
                        temporary_directory=root / "temporary",
                        plot_sample_rows=3,
                        plot_sample_seed=2309,
                        score_run_rows=2,
                    )
                    """
                )
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(
                    (Path(__file__).resolve().parents[1] / "src").resolve()
                )
                child = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(child.returncode, exit_code, child.stderr)

                journal = root / ".metrics.json.evaluation-transaction.json"
                self.assertEqual(
                    json.loads(journal.read_text(encoding="utf-8"))["state"],
                    "building",
                )
                work_parent = root / "temporary"
                work = list(work_parent.glob("evaluation-*"))
                self.assertEqual(len(work), 1)
                entries = list(work[0].iterdir())
                if mode == "after_mkdir":
                    self.assertEqual(entries, [])
                else:
                    self.assertEqual(len(entries), 1)
                    self.assertIn("initializing", entries[0].name)

                foreign = work_parent / "foreign-sibling.txt"
                foreign.write_text("preserve", encoding="utf-8")
                recovered = self._evaluate(root, (1, 3))
                self.assertEqual(recovered["rows"], 5)
                self.assertEqual(list(work_parent.glob("evaluation-*")), [])
                self.assertEqual(foreign.read_text(encoding="utf-8"), "preserve")

    def test_temp_parent_cleanup_respects_creation_and_emptiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created_parent = root / "created-temporary"
            self._evaluate(root, (1, 3), temporary_directory=created_parent)
            self.assertFalse(created_parent.exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preexisting_parent = root / "preexisting-temporary"
            preexisting_parent.mkdir()
            self._evaluate(root, (1, 3), temporary_directory=preexisting_parent)
            self.assertTrue(preexisting_parent.is_dir())
            self.assertEqual(list(preexisting_parent.iterdir()), [])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created_parent = root / "created-temporary"

            def fail_build(_index: int, _path: Path) -> None:
                raise RuntimeError("simulated build exception")

            with (
                patch.object(
                    evaluation_module,
                    "_after_build_boundary",
                    side_effect=fail_build,
                ),
                self.assertRaisesRegex(RuntimeError, "simulated build exception"),
            ):
                self._evaluate(root, (1, 3), temporary_directory=created_parent)
            self.assertFalse(created_parent.exists())

    def test_success_preserves_a_late_foreign_temporary_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            foreign = root / "temporary" / "foreign-sibling.txt"

            def add_foreign_sibling(index: int, _path: Path) -> None:
                if index == 0:
                    foreign.write_text("preserve", encoding="utf-8")

            with patch.object(
                evaluation_module,
                "_after_build_boundary",
                side_effect=add_foreign_sibling,
                create=True,
            ):
                result = self._evaluate(root, (1, 3))
            self.assertEqual(result["rows"], 5)
            self.assertEqual(foreign.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(list((root / "temporary").glob("evaluation-*")), [])

    def test_transaction_lock_rejects_a_concurrent_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = evaluation_module._transaction_path(root / "metrics.json")
            lock = evaluation_module._acquire_transaction_lock(
                evaluation_module._transaction_lock_path(journal)
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "another evaluation process"):
                    self._evaluate(root, (1, 3))
            finally:
                evaluation_module._release_transaction_lock(lock)

    def test_parquet_replay_keeps_state_across_batches_and_restores_storage_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            self._evaluate(root, (1, 3))
            config = deepcopy(DEFAULTS)
            config["temporal"]["enabled"] = True
            expected_frame = pq.read_table(root / "predictions.parquet").to_pandas()
            expected_temporal, expected_metrics = replay_predictions(expected_frame, config)
            with patch("bitguard_bnn.out_of_core.replay._MAX_MERGE_FAN_IN", 2):
                actual_metrics = replay_parquet_predictions(
                    root / "predictions.parquet",
                    root / "temporal_predictions.parquet",
                    config,
                    temporary_directory=root / "replay-temporary",
                    batch_rows=2,
                    order_run_rows=1,
                )
            actual = pq.read_table(root / "temporal_predictions.parquet").to_pandas()
            self.assertEqual(actual["row_uid"].tolist(), expected_frame["row_uid"].tolist())
            columns = ["row_uid", "risk_score", "action_level", "stateful_predicted_label"]
            pd.testing.assert_frame_equal(
                actual[columns].sort_values("row_uid").reset_index(drop=True),
                expected_temporal[columns].sort_values("row_uid").reset_index(drop=True),
                check_dtype=False,
            )
            self._assert_nested_close(expected_metrics, actual_metrics)
            self.assertFalse((root / "replay-temporary").exists())

    def test_parquet_replay_uses_original_position_for_equal_order_keys(self) -> None:
        rows = {
            "row_uid": np.asarray(["z-first", "a-second", "m-third"]),
            "true_label": np.asarray(["scan_like", "benign", "scan_like"]),
            "predicted_label": np.asarray(["scan_like", "benign", "scan_like"]),
            "probabilities": np.asarray(
                [[0.05, 0.9, 0.05], [0.95, 0.025, 0.025], [0.05, 0.9, 0.05]],
                dtype=np.float32,
            ),
            "exit_stage": np.asarray([2, 0, 2], dtype=np.int16),
            "source_file": np.asarray(["episode"] * 3),
            "device_id": np.asarray(["device"] * 3),
            "timestamp": np.asarray([1.0, 1.0, 2.0]),
            "sequence_index": np.asarray([1, 1, 2], dtype=np.int64),
            "raw_attack": np.asarray(["scan", "benign", "scan"]),
            "has_wall_clock_time": np.asarray([True] * 3),
            "temporal_continuity": np.asarray([True] * 3),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._evaluate_rows(root, rows, test_contract="tie-order-contract-v1")
            config = deepcopy(DEFAULTS)
            config["temporal"]["enabled"] = True
            frame = pq.read_table(root / "predictions.parquet").to_pandas()
            expected_frame, expected_metrics = replay_predictions(frame, config)
            actual_metrics = replay_parquet_predictions(
                root / "predictions.parquet",
                root / "temporal_predictions.parquet",
                config,
                temporary_directory=root / "replay-temporary",
                batch_rows=1,
                order_run_rows=1,
            )
            actual_frame = pq.read_table(root / "temporal_predictions.parquet").to_pandas()
            compare = ["row_uid", "risk_score", "action_level", "stateful_predicted_label"]
            pd.testing.assert_frame_equal(
                expected_frame[compare].sort_values("row_uid").reset_index(drop=True),
                actual_frame[compare].sort_values("row_uid").reset_index(drop=True),
                check_dtype=False,
            )
            self._assert_nested_close(expected_metrics, actual_metrics)

    def test_parquet_replay_withholds_wall_clock_metrics_when_continuity_is_false(self) -> None:
        rows = self._rows()
        rows["temporal_continuity"] = np.asarray([True, True, False, True, True])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._evaluate_rows(root, rows, test_contract="discontinuous-contract-v1")
            config = deepcopy(DEFAULTS)
            config["temporal"]["enabled"] = True
            frame = pq.read_table(root / "predictions.parquet").to_pandas()
            _expected_frame, expected_metrics = replay_predictions(frame, config)
            actual_metrics = replay_parquet_predictions(
                root / "predictions.parquet",
                root / "temporal_predictions.parquet",
                config,
                temporary_directory=root / "replay-temporary",
                batch_rows=2,
                order_run_rows=2,
            )
            self._assert_nested_close(expected_metrics, actual_metrics)
            self.assertIsNone(actual_metrics["detection_delay_seconds_p50"])
            self.assertEqual(actual_metrics["time_to_mitigation_unit"], "decisions")
            self.assertIsNone(actual_metrics["observed_device_hours"])

    def test_replay_publish_never_clobbers_or_deletes_foreign_destination(self) -> None:
        config = deepcopy(DEFAULTS)
        config["temporal"]["enabled"] = True
        for mode in ("before_publish", "after_publish"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._evaluate(root, (1, 3))
                destination = root / "temporal_predictions.parquet"
                temporary = root / "replay-temporary"

                def foreign_before(_destination: Path, _partial: Path) -> None:
                    destination.write_bytes(b"foreign-before-publish")

                def foreign_after(_destination: Path, _identity: object) -> None:
                    destination.unlink()
                    destination.write_bytes(b"foreign-after-publish")

                hook = (
                    patch.object(
                        replay_module,
                        "_before_replay_publish",
                        side_effect=foreign_before,
                        create=True,
                    )
                    if mode == "before_publish"
                    else patch.object(
                        replay_module,
                        "_after_replay_publish",
                        side_effect=foreign_after,
                        create=True,
                    )
                )
                expected_error = FileExistsError if mode == "before_publish" else RuntimeError
                with hook, self.assertRaises(expected_error):
                    replay_parquet_predictions(
                        root / "predictions.parquet",
                        destination,
                        config,
                        temporary_directory=temporary,
                        batch_rows=2,
                        order_run_rows=1,
                    )
                self.assertEqual(
                    destination.read_bytes(),
                    (
                        b"foreign-before-publish"
                        if mode == "before_publish"
                        else b"foreign-after-publish"
                    ),
                )
                self.assertFalse(temporary.exists())

    def test_replay_construction_failure_cleans_owned_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._evaluate(root, (1, 3))
            temporary = root / "replay-temporary"
            destination_parent = root / "new-output-parent"
            with (
                patch.object(
                    replay_module,
                    "_OperationalAccumulator",
                    side_effect=OSError("injected sqlite construction failure"),
                ),
                self.assertRaisesRegex(OSError, "injected sqlite construction failure"),
            ):
                replay_parquet_predictions(
                    root / "predictions.parquet",
                    destination_parent / "temporal_predictions.parquet",
                    deepcopy(DEFAULTS),
                    temporary_directory=temporary,
                    batch_rows=2,
                    order_run_rows=1,
                )
            self.assertFalse(temporary.exists())
            self.assertFalse(destination_parent.exists())

    def test_replay_success_preserves_late_foreign_temporary_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._evaluate(root, (1, 3))
            temporary = root / "replay-temporary"
            foreign = temporary / "foreign-sibling.txt"

            def add_foreign(_destination: Path, _partial: Path) -> None:
                foreign.write_text("preserve", encoding="utf-8")

            with patch.object(
                replay_module,
                "_before_replay_publish",
                side_effect=add_foreign,
                create=True,
            ):
                metrics = replay_parquet_predictions(
                    root / "predictions.parquet",
                    root / "temporal_predictions.parquet",
                    deepcopy(DEFAULTS),
                    temporary_directory=temporary,
                    batch_rows=2,
                    order_run_rows=1,
                )
            self.assertEqual(metrics["rows"], 5)
            self.assertEqual(foreign.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(list(temporary.glob("replay-*")), [])

    def test_replay_process_exit_build_and_post_publish_are_recoverable(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(
            (Path(__file__).resolve().parents[1] / "src").resolve()
        )
        for mode, exit_code in (
            ("build", 95),
            ("linked_before_unlink", 97),
            ("post_publish", 96),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._evaluate(root, (1, 3))
                destination = root / "temporal_predictions.parquet"
                temporary = root / "replay-temporary"
                if mode == "build":
                    hook = (
                        "module._after_replay_build_boundary = "
                        f"lambda index, path: os._exit({exit_code})"
                    )
                elif mode == "linked_before_unlink":
                    hook = (
                        "module._after_replay_link_boundary = "
                        f"lambda destination, partial: os._exit({exit_code})"
                    )
                else:
                    hook = (
                        "module._after_replay_publish = "
                        f"lambda path, identity: os._exit({exit_code})"
                    )
                script = textwrap.dedent(
                    f"""
                    import os
                    from copy import deepcopy
                    from pathlib import Path
                    import bitguard_bnn.out_of_core.replay as module
                    from bitguard_bnn.config import DEFAULTS

                    root = Path({str(root)!r})
                    {hook}
                    module.replay_parquet_predictions(
                        root / "predictions.parquet",
                        root / "temporal_predictions.parquet",
                        deepcopy(DEFAULTS),
                        temporary_directory=root / "replay-temporary",
                        batch_rows=2,
                        order_run_rows=1,
                    )
                    """
                )
                child = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(child.returncode, exit_code, child.stderr)
                journal = root / ".temporal_predictions.parquet.replay-transaction.json"
                self.assertEqual(
                    json.loads(journal.read_text(encoding="utf-8"))["state"],
                    "building" if mode == "build" else "publishing",
                )
                foreign = temporary / "foreign-sibling.txt"
                foreign.write_text("preserve", encoding="utf-8")

                if mode == "build":
                    build_boundaries: list[int] = []

                    def observed_build(index: int, _path: Path) -> None:
                        build_boundaries.append(index)

                    with patch.object(
                        replay_module,
                        "_after_replay_build_boundary",
                        side_effect=observed_build,
                    ):
                        recovered = replay_parquet_predictions(
                            root / "predictions.parquet",
                            destination,
                            deepcopy(DEFAULTS),
                            temporary_directory=temporary,
                            batch_rows=2,
                            order_run_rows=1,
                        )
                    self.assertTrue(build_boundaries)
                else:
                    with patch.object(
                        replay_module.pq,
                        "ParquetFile",
                        side_effect=AssertionError("recovery must not replay input"),
                    ) as parquet_file:
                        recovered = replay_parquet_predictions(
                            root / "predictions.parquet",
                            destination,
                            deepcopy(DEFAULTS),
                            temporary_directory=temporary,
                            batch_rows=2,
                            order_run_rows=1,
                        )
                    parquet_file.assert_not_called()

                self.assertEqual(recovered["rows"], 5)
                self.assertTrue(destination.is_file())
                self.assertEqual(foreign.read_text(encoding="utf-8"), "preserve")
                self.assertEqual(list(temporary.glob("replay-*")), [])
                self.assertEqual(
                    json.loads(journal.read_text(encoding="utf-8"))["state"],
                    "committed",
                )
                with patch.object(
                    replay_module.pq,
                    "ParquetFile",
                    side_effect=AssertionError("committed replay must be reused"),
                ) as parquet_file:
                    reused = replay_parquet_predictions(
                        root / "predictions.parquet",
                        destination,
                        deepcopy(DEFAULTS),
                        temporary_directory=temporary,
                        batch_rows=2,
                        order_run_rows=1,
                    )
                parquet_file.assert_not_called()
                self._assert_nested_close(recovered, reused)

    def test_replay_closes_parquet_on_construction_and_lock_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._evaluate(root, (1, 3))
            destination = root / "temporal_predictions.parquet"
            real_parquet = pq.ParquetFile(root / "predictions.parquet")
            close_calls: list[bool] = []

            class TrackingParquet:
                schema_arrow = real_parquet.schema_arrow

                def iter_batches(self, *args: object, **kwargs: object):
                    return real_parquet.iter_batches(*args, **kwargs)

                def close(self, *, force: bool = False) -> None:
                    close_calls.append(force)
                    real_parquet.close(force=force)

            with (
                patch.object(replay_module.pq, "ParquetFile", return_value=TrackingParquet()),
                patch.object(
                    replay_module,
                    "_OperationalAccumulator",
                    side_effect=OSError("injected accumulator failure"),
                ),
                self.assertRaisesRegex(OSError, "injected accumulator failure"),
            ):
                replay_parquet_predictions(
                    root / "predictions.parquet",
                    destination,
                    deepcopy(DEFAULTS),
                    temporary_directory=root / "replay-temporary",
                    batch_rows=2,
                    order_run_rows=1,
                )
            self.assertEqual(close_calls, [True])

            lock = replay_module._acquire_replay_lock(destination)
            try:
                with (
                    patch.object(
                        replay_module.pq,
                        "ParquetFile",
                        side_effect=AssertionError("Parquet opened before lock"),
                    ) as parquet_file,
                    self.assertRaisesRegex(RuntimeError, "another replay process"),
                ):
                    replay_parquet_predictions(
                        root / "predictions.parquet",
                        destination,
                        deepcopy(DEFAULTS),
                        temporary_directory=root / "replay-temporary",
                    )
                parquet_file.assert_not_called()
            finally:
                replay_module._release_replay_lock(lock)

    @unittest.skipUnless(os.name == "nt", "NTFS case-alias lock contract")
    def test_replay_lock_normalizes_windows_case_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upper = root / "Temporal-Predictions.parquet"
            lower = root / "temporal-predictions.parquet"
            self.assertEqual(
                replay_module._replay_lock_path(upper),
                replay_module._replay_lock_path(lower),
            )

    def test_legacy_delimiter_collision_pairs_keep_independent_csv_and_parquet_state(self) -> None:
        config = deepcopy(DEFAULTS)
        config["temporal"]["enabled"] = True
        first = ("source::device", "tail")
        second = ("source", "device::tail")
        self.assertEqual(f"{first[0]}::{first[1]}", f"{second[0]}::{second[1]}")
        self.assertNotEqual(temporal_state_key(*first), temporal_state_key(*second))
        machine = TemporalSecurityStateMachine(config)
        probability_columns = ["prob_benign", "prob_scan_like", "prob_unknown_like"]
        for source, device in (first, second):
            replay_prediction_row(
                {
                    "source_file": source,
                    "device_id": device,
                    "true_label": "scan_like",
                    "predicted_label": "scan_like",
                    "prob_benign": 0.1,
                    "prob_scan_like": 0.8,
                    "prob_unknown_like": 0.1,
                    "timestamp": 1.0,
                },
                machine,
                probability_columns,
            )
        self.assertEqual(len(machine.states), 2)

        rows = {
            "row_uid": np.asarray(["collision-a1", "collision-b1", "collision-a2", "collision-b2"]),
            "true_label": np.asarray(["scan_like", "benign", "scan_like", "benign"]),
            "predicted_label": np.asarray(["scan_like", "benign", "scan_like", "benign"]),
            "probabilities": np.asarray(
                [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05]],
                dtype=np.float32,
            ),
            "exit_stage": np.asarray([2, 0, 2, 0], dtype=np.int16),
            "source_file": np.asarray([first[0], second[0], first[0], second[0]]),
            "device_id": np.asarray([first[1], second[1], first[1], second[1]]),
            "timestamp": np.asarray([1.0, 2.0, 3.0, 4.0]),
            "sequence_index": np.asarray([1, 1, 2, 2]),
            "raw_attack": np.asarray(["scan", "benign", "scan", "benign"]),
            "has_wall_clock_time": np.asarray([True] * 4),
            "temporal_continuity": np.asarray([True] * 4),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluate_prediction_batches(
                lambda: iter([{key: value[:2] for key, value in rows.items()}, {key: value[2:] for key, value in rows.items()}]),
                probability_labels=self.labels,
                high_risk_labels=["scan_like", "unknown_like"],
                test_contract="collision-test-contract-v1",
                prediction_path=root / "predictions.parquet",
                metrics_path=root / "metrics.json",
                plot_sample_path=root / "sample.parquet",
                plot_manifest_path=root / "sample.json",
                temporary_directory=root / "evaluation-temporary",
                plot_sample_rows=2,
            )
            frame = pq.read_table(root / "predictions.parquet").to_pandas()
            expected_frame, expected_metrics = replay_predictions(frame, config)
            actual_metrics = replay_parquet_predictions(
                root / "predictions.parquet",
                root / "temporal_predictions.parquet",
                config,
                temporary_directory=root / "replay-temporary",
                batch_rows=1,
                order_run_rows=1,
            )
            actual_frame = pq.read_table(root / "temporal_predictions.parquet").to_pandas()
            self._assert_nested_close(expected_metrics, actual_metrics)
            compare = ["row_uid", "risk_score", "action_level", "stateful_predicted_label"]
            pd.testing.assert_frame_equal(
                expected_frame[compare].sort_values("row_uid").reset_index(drop=True),
                actual_frame[compare].sort_values("row_uid").reset_index(drop=True),
                check_dtype=False,
            )

    def test_cli_dispatches_parquet_before_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "predictions.parquet").touch()
            config = deepcopy(DEFAULTS)
            with (
                patch("bitguard_bnn.config.load_config", return_value=config),
                patch(
                    "bitguard_bnn.out_of_core.replay.replay_parquet_predictions",
                    return_value={"rows": 5},
                ) as replay,
            ):
                main(["replay", "--run", str(run)])
            replay.assert_called_once()

    def test_cli_keeps_csv_replay_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            frame = pd.DataFrame(
                {
                    "device_id": ["device"],
                    "true_label": ["benign"],
                    "predicted_label": ["benign"],
                    "prob_benign": [1.0],
                }
            )
            frame.to_csv(run / "predictions.csv", index=False)
            config = deepcopy(DEFAULTS)
            temporal = frame.assign(action_level=0)
            with (
                patch("bitguard_bnn.config.load_config", return_value=config),
                patch(
                    "bitguard_bnn.state.replay_predictions",
                    return_value=(temporal, {"rows": 1}),
                ) as replay,
            ):
                main(["replay", "--run", str(run)])
            replay.assert_called_once()
            self.assertTrue((run / "temporal_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
