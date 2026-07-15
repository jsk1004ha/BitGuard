from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.out_of_core.preprocess import (
    ClassSufficientStatistics,
    StreamingFeaturePreprocessor,
)
from bitguard_bnn.out_of_core.quantiles import PriorityRowSketch
from bitguard_bnn.preprocess import FeaturePreprocessor


def _config(
    *,
    selection: str = "f_score",
    scaler: str = "robust",
    encoder: str = "sign",
    feature_budget: int | None = 2,
    bits: int = 2,
) -> dict[str, object]:
    config = copy.deepcopy(DEFAULTS)
    config["_project_root"] = "."
    config["_config_path"] = "config.yaml"
    config["preprocess"].update(
        {
            "selection": selection,
            "scaler": scaler,
            "encoder": encoder,
            "feature_budget": feature_budget,
            "thermometer_bits": bits,
            "feature_cost_csv": None,
        }
    )
    config["preprocess"]["open_set"].update(
        {
            "enabled": True,
            "confidence_threshold": 0.6,
            "benign_distance_quantile": 0.9,
        }
    )
    return config


def _frame(rows: int = 120, *, missing: bool = False) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(20260715)
    labels = np.asarray(
        ["benign", "scan_like", "flood_like"] * (rows // 3), dtype=object
    )
    if len(labels) < rows:
        labels = np.concatenate([labels, labels[: rows - len(labels)]])
    class_index = np.asarray(
        [{"benign": 0, "scan_like": 1, "flood_like": 2}[str(label)] for label in labels]
    )
    values = np.column_stack(
        [
            class_index + rng.normal(0.0, 0.2, rows),
            rng.normal(0.0, 1.0, rows),
            0.4 * class_index + rng.normal(0.0, 0.2, rows),
            np.ones(rows),
        ]
    )
    if missing:
        values[::9, 0] = np.nan
        values[::7, 1] = np.inf
        values[5, 2] = 10_000.0
    features = ["signal", "noise", "secondary", "constant"]
    frame = pd.DataFrame(values, columns=features)
    frame["row_uid"] = [f"row-{index:06d}" for index in range(rows)]
    frame["behavior_label"] = labels
    return frame, features


def _feed(
    builder: StreamingFeaturePreprocessor,
    method: str,
    frame: pd.DataFrame,
    features: list[str],
    indices: np.ndarray,
    *,
    membership: str = "train",
    split_fingerprint: str = "split-v1",
) -> None:
    function = getattr(builder, method)
    for chunk in np.array_split(indices, 5):
        if not len(chunk):
            continue
        part = frame.iloc[chunk]
        function(
            part["row_uid"].astype(str).to_numpy(),
            part[features].to_numpy(),
            part["behavior_label"].astype(str).to_numpy(),
            split_fingerprint=split_fingerprint,
            feature_names=features,
            membership=np.full(len(part), membership, dtype=object),
        )


def _fit_streaming(
    frame: pd.DataFrame,
    features: list[str],
    config: dict[str, object],
    *,
    capacity: int | None = None,
) -> tuple[FeaturePreprocessor, StreamingFeaturePreprocessor]:
    builder = StreamingFeaturePreprocessor(
        config,
        candidate_features=features,
        split_fingerprint="split-v1",
        expected_train_rows=len(frame),
        quantile_capacity=capacity or max(len(frame), 4),
        quantile_seed=17,
    )
    forward = np.arange(len(frame))
    reverse = forward[::-1]
    odd_even = np.concatenate([forward[::2], forward[1::2]])
    _feed(builder, "inspect_batch", frame, features, odd_even)
    builder.finalize_imputation()
    _feed(builder, "accumulate_anova_batch", frame, features, reverse)
    builder.finalize_selection()
    _feed(builder, "calibrate_selected_batch", frame, features, forward)
    return builder.finalize(), builder


def _artifact_signature(processor: FeaturePreprocessor) -> str:
    payload: dict[str, object] = {
        "manifest": processor.feature_manifest(),
        "imputer": np.asarray(processor.imputer.statistics_).tolist(),
        "selected": processor.selected_features,
        "scores": np.asarray(processor.selection_scores).tolist(),
        "costs": np.asarray(processor.feature_costs).tolist(),
        "encoder": (
            None
            if processor.encoder.thresholds is None
            else np.asarray(processor.encoder.thresholds).tolist()
        ),
        "center": np.asarray(processor.benign_center).tolist(),
        "distance": processor.open_distance_threshold,
    }
    for name in ("center_", "mean_", "var_", "scale_"):
        value = getattr(processor.scaler, name, None)
        payload[f"scaler_{name}"] = None if value is None else np.asarray(value).tolist()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class OutOfCorePreprocessTests(unittest.TestCase):
    def test_no_missing_selection_matches_in_memory_preprocessor(self) -> None:
        frame, features = _frame()
        config = _config(feature_budget=3)
        expected = FeaturePreprocessor(copy.deepcopy(config)).fit(frame, features)

        actual, _ = _fit_streaming(frame, features, copy.deepcopy(config))

        self.assertEqual(actual.selected_features, expected.selected_features)
        np.testing.assert_allclose(
            actual.selection_scores,
            expected.selection_scores,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual.transform_unencoded(frame),
            expected.transform_unencoded(frame),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_missing_and_outlier_quantiles_have_declared_fixture_tolerance(self) -> None:
        frame, features = _frame(300, missing=True)
        config = _config(feature_budget=3)
        expected = FeaturePreprocessor(copy.deepcopy(config)).fit(frame, features)

        actual, _ = _fit_streaming(
            frame,
            features,
            copy.deepcopy(config),
            capacity=128,
        )

        self.assertEqual(actual.selected_features, expected.selected_features)
        np.testing.assert_allclose(
            actual.imputer.statistics_, expected.imputer.statistics_, rtol=0.2, atol=0.2
        )
        self.assertTrue(np.isfinite(actual.transform(frame)).all())

    def test_mergeable_anova_statistics_match_sklearn(self) -> None:
        frame, features = _frame(90)
        values = frame[features].to_numpy(np.float64)
        labels = frame["behavior_label"].astype(str).to_numpy()
        active = ("benign", "scan_like", "flood_like")
        whole = ClassSufficientStatistics(active, len(features))
        left = ClassSufficientStatistics(active, len(features))
        right = ClassSufficientStatistics(active, len(features))
        whole.update(values, labels)
        left.update(values[::2], labels[::2])
        right.update(values[1::2], labels[1::2])
        left.merge(right)
        y = np.asarray([active.index(label) for label in labels])
        expected, _ = f_classif(values, y)
        expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)

        np.testing.assert_allclose(whole.finalize_f_scores(), expected, rtol=1e-10)
        np.testing.assert_allclose(left.finalize_f_scores(), expected, rtol=1e-10)

    def test_constant_ties_and_n_equal_k_are_stable_zeros(self) -> None:
        frame = pd.DataFrame(
            {
                "first": [1.0, 1.0],
                "second": [2.0, 2.0],
                "row_uid": ["a", "b"],
                "behavior_label": ["benign", "scan_like"],
            }
        )
        processor, _ = _fit_streaming(
            frame,
            ["first", "second"],
            _config(feature_budget=1),
        )
        self.assertEqual(processor.selected_features, ["first"])
        self.assertIsNotNone(processor.selection_scores)
        assert processor.selection_scores is not None
        self.assertEqual(processor.selection_scores.tolist(), [0.0])

        one_class = frame.copy()
        one_class["behavior_label"] = "benign"
        builder = StreamingFeaturePreprocessor(
            _config(),
            candidate_features=["first", "second"],
            split_fingerprint="split-v1",
            expected_train_rows=2,
            quantile_capacity=2,
            quantile_seed=1,
        )
        _feed(builder, "inspect_batch", one_class, ["first", "second"], np.arange(2))
        with self.assertRaisesRegex(ValueError, "two known"):
            builder.finalize_imputation()

    def test_scaler_and_encoder_modes_hydrate_transform_contracts(self) -> None:
        frame, features = _frame(60)
        cases = (
            ("standard", "none", 2),
            ("robust", "sign", 2),
            ("none", "thermometer", 3),
            ("standard", "hybrid", 2),
        )
        for scaler, encoder, bits in cases:
            with self.subTest(scaler=scaler, encoder=encoder):
                processor, _ = _fit_streaming(
                    frame,
                    features,
                    _config(scaler=scaler, encoder=encoder, bits=bits),
                )
                transformed = processor.transform(frame)
                expected_width = (
                    len(processor.selected_features) * bits
                    if encoder == "thermometer"
                    else len(processor.selected_features)
                )
                self.assertEqual(transformed.shape, (len(frame), expected_width))
                self.assertTrue(np.isfinite(transformed).all())
                self.assertEqual(processor.imputer.n_features_in_, len(features))
                self.assertEqual(processor.scaler.n_features_in_, 2)

    def test_joblib_roundtrip_preserves_transform_and_export_attributes(self) -> None:
        frame, features = _frame(60, missing=True)
        processor, _ = _fit_streaming(frame, features, _config())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "preprocessor.joblib"
            processor.save(path)
            restored = FeaturePreprocessor.load(path)

        np.testing.assert_array_equal(restored.transform(frame), processor.transform(frame))
        np.testing.assert_array_equal(restored.imputer.statistics_, processor.imputer.statistics_)
        self.assertTrue(hasattr(restored.scaler, "scale_"))
        self.assertIsNotNone(restored.benign_center)
        self.assertGreater(restored.open_distance_threshold, 0.0)

    def test_state_machine_rejects_nontrain_unknown_duplicate_and_wrong_proof(self) -> None:
        frame, features = _frame(12)
        builder = StreamingFeaturePreprocessor(
            _config(),
            candidate_features=features,
            split_fingerprint="split-v1",
            expected_train_rows=len(frame),
            quantile_capacity=12,
            quantile_seed=1,
        )
        with self.assertRaisesRegex(RuntimeError, "inspect"):
            _feed(builder, "accumulate_anova_batch", frame, features, np.arange(len(frame)))
        for membership, fingerprint in (("validation", "split-v1"), ("train", "wrong")):
            with self.subTest(membership=membership, fingerprint=fingerprint), self.assertRaises(
                ValueError
            ):
                _feed(
                    builder,
                    "inspect_batch",
                    frame,
                    features,
                    np.arange(2),
                    membership=membership,
                    split_fingerprint=fingerprint,
                )
        unknown = frame.iloc[:1].copy()
        unknown["behavior_label"] = "unknown_like"
        with self.assertRaisesRegex(ValueError, "unknown_like"):
            _feed(builder, "inspect_batch", unknown, features, np.arange(1))
        duplicate = frame.iloc[[0, 0]].copy()
        with self.assertRaisesRegex(ValueError, "duplicate row_uid"):
            _feed(builder, "inspect_batch", duplicate, features, np.arange(2))

    def test_pass_mismatch_is_rejected(self) -> None:
        frame, features = _frame(30)
        builder = StreamingFeaturePreprocessor(
            _config(),
            candidate_features=features,
            split_fingerprint="split-v1",
            expected_train_rows=len(frame),
            quantile_capacity=30,
            quantile_seed=1,
        )
        _feed(builder, "inspect_batch", frame, features, np.arange(len(frame)))
        builder.finalize_imputation()
        changed = frame.copy()
        changed.loc[3, "signal"] += 1.0
        _feed(builder, "accumulate_anova_batch", changed, features, np.arange(len(frame)))
        with self.assertRaisesRegex(ValueError, "pass.*mismatch|changed"):
            builder.finalize_selection()

    def test_rejected_validation_mutation_cannot_change_artifact_but_train_can(self) -> None:
        frame, features = _frame(60)
        baseline, _ = _fit_streaming(frame, features, _config())
        mutated = frame.copy()
        mutated.loc[:, features] += 100.0
        mutated["row_uid"] = "validation-" + mutated["row_uid"].astype(str)
        mutated["behavior_label"] = "unknown_like"

        builder = StreamingFeaturePreprocessor(
            _config(),
            candidate_features=features,
            split_fingerprint="split-v1",
            expected_train_rows=len(frame),
            quantile_capacity=60,
            quantile_seed=17,
        )
        with self.assertRaises(ValueError):
            _feed(
                builder,
                "inspect_batch",
                mutated,
                features,
                np.arange(len(mutated)),
                membership="validation",
            )
        _feed(builder, "inspect_batch", frame, features, np.arange(len(frame)))
        builder.finalize_imputation()
        _feed(builder, "accumulate_anova_batch", frame, features, np.arange(len(frame)))
        builder.finalize_selection()
        _feed(builder, "calibrate_selected_batch", frame, features, np.arange(len(frame)))
        after_rejection = builder.finalize()
        self.assertEqual(_artifact_signature(after_rejection), _artifact_signature(baseline))

        changed_train = frame.copy()
        changed_train.loc[:, "signal"] += np.linspace(0.0, 20.0, len(frame))
        changed_artifact, _ = _fit_streaming(changed_train, features, _config())
        self.assertNotEqual(
            _artifact_signature(changed_artifact), _artifact_signature(baseline)
        )

    def test_sparse_feature_without_finite_retained_sample_fails_explicitly(self) -> None:
        uids = [f"uid-{index}" for index in range(8)]
        probe = PriorityRowSketch(capacity=1, seed=7, width=1)
        probe.update_many(uids, np.zeros((len(uids), 1)))
        retained_uid = probe.retained_rows()[0][0]
        finite_uid = next(uid for uid in uids if uid != retained_uid)
        frame = pd.DataFrame(
            {
                "sparse": [1.0 if uid == finite_uid else np.nan for uid in uids],
                "signal": np.arange(len(uids), dtype=float),
                "row_uid": uids,
                "behavior_label": ["benign", "scan_like"] * 4,
            }
        )
        builder = StreamingFeaturePreprocessor(
            _config(),
            candidate_features=["sparse", "signal"],
            split_fingerprint="split-v1",
            expected_train_rows=len(frame),
            quantile_capacity=1,
            quantile_seed=7,
        )
        _feed(
            builder,
            "inspect_batch",
            frame,
            ["sparse", "signal"],
            np.arange(len(frame)),
        )
        with self.assertRaisesRegex(ValueError, "finite retained sample.*sparse"):
            builder.finalize_imputation()

    def test_bounded_state_and_streaming_provenance(self) -> None:
        frame, features = _frame(120)
        processor, builder = _fit_streaming(frame, features, _config(), capacity=7)
        self.assertLessEqual(builder.imputation_sketch.retained_count, 7)
        self.assertIsNotNone(builder.selected_calibration)
        assert builder.selected_calibration is not None
        self.assertLessEqual(builder.selected_calibration.retained_count, 7)
        self.assertFalse(any(isinstance(value, set) for value in vars(builder).values()))
        manifest = processor.feature_manifest()
        self.assertEqual(manifest["fit_mode"], "streaming_priority_sketch")
        provenance = manifest["fit_provenance"]
        self.assertEqual(provenance["split_fingerprint"], "split-v1")
        self.assertEqual(provenance["rows_considered"], 120)
        self.assertEqual(provenance["passes"], 3)
        self.assertIn("approximate_fields", provenance)
        self.assertIn("exact_fields", provenance)

    def test_in_memory_manifest_declares_exact_fit_mode(self) -> None:
        frame, features = _frame(30)
        processor = FeaturePreprocessor(_config()).fit(frame, features)
        manifest = processor.feature_manifest()
        self.assertEqual(manifest["fit_mode"], "in_memory_exact")
        self.assertEqual(manifest["fit_provenance"]["fit_mode"], "in_memory_exact")


if __name__ == "__main__":
    unittest.main()
