from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bitguard_bnn.rbc_encoder import RiskWeightedComparisonEncoder


class RiskWeightedComparisonEncoderTest(unittest.TestCase):
    @staticmethod
    def collision_fixture() -> tuple[pd.DataFrame, np.ndarray]:
        rng = np.random.default_rng(9)
        labels = np.asarray(["benign"] * 80 + ["attack"] * 80)
        return (
            pd.DataFrame(
                {
                    "uninformative": rng.normal(size=160),
                    "boundary": np.r_[
                        rng.normal(-2.0, 0.15, 80),
                        rng.normal(2.0, 0.15, 80),
                    ],
                    "device": ["sensor"] * 160,
                }
            ),
            labels,
        )

    def test_fixed_budget_uses_only_numeric_features_and_reduces_collision(self) -> None:
        frame, labels = self.collision_fixture()
        encoder = RiskWeightedComparisonEncoder(
            1,
            false_negative_cost=4.0,
            selection_fraction=0.3,
            random_state=7,
        ).fit(frame, labels)

        encoded = encoder.transform(frame)
        self.assertEqual(encoded.shape, (len(frame), 1))
        self.assertEqual(set(np.unique(encoded)), {-1.0, 1.0})
        self.assertEqual(encoder.feature_names_, ["uninformative", "boundary"])
        self.assertEqual(encoder.comparisons_[0].feature, "boundary")
        self.assertLess(encoder.selection_collision_, encoder.initial_selection_collision_)
        self.assertTrue(encoder.replacement_history_)

    def test_missing_values_use_search_only_median_deterministically(self) -> None:
        frame, labels = self.collision_fixture()
        frame.loc[[0, 3, 81], "boundary"] = np.nan
        first = RiskWeightedComparisonEncoder(2, random_state=13).fit(frame, labels)
        second = RiskWeightedComparisonEncoder(2, random_state=13).fit(frame, labels)
        probe = pd.DataFrame(
            {"uninformative": [np.nan], "boundary": [np.nan], "device": ["new"]}
        )

        np.testing.assert_array_equal(first.transform(probe), second.transform(probe))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(np.isfinite(list(first.imputation_values_.values())).all())

    def test_raw_feature_cost_cap_applies_to_unique_features(self) -> None:
        frame, labels = self.collision_fixture()
        encoder = RiskWeightedComparisonEncoder(
            4,
            raw_feature_cost_cap=1.0,
            random_state=5,
        ).fit(
            frame,
            labels,
            feature_costs={"uninformative": 1.0, "boundary": 1.0},
        )

        self.assertEqual(len(encoder.comparisons_), 4)
        self.assertLessEqual(encoder.raw_feature_cost, 1.0)
        self.assertEqual(len({item.feature for item in encoder.comparisons_}), 1)

    def test_raw_thresholds_and_json_round_trip_preserve_output(self) -> None:
        frame, labels = self.collision_fixture()
        encoder = RiskWeightedComparisonEncoder(3, random_state=3).fit(frame, labels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rbc_encoder.json"
            encoder.save(path)
            restored = RiskWeightedComparisonEncoder.load(path)

        np.testing.assert_array_equal(encoder.transform(frame), restored.transform(frame))
        self.assertEqual(restored.encoded_dimension, 3)
        self.assertTrue(
            all(item["operator"] == ">=" for item in restored.to_dict()["comparisons"])
        )
        self.assertEqual(restored.to_dict()["raw_feature_cost"], restored.raw_feature_cost)

    def test_fit_requires_enough_rows_for_internal_stratified_selection(self) -> None:
        frame = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        with self.assertRaisesRegex(ValueError, "at least two benign and two attack"):
            RiskWeightedComparisonEncoder(1).fit(
                frame, np.asarray(["benign", "attack", "attack"])
            )

    def test_boolean_targets_and_group_disjoint_internal_selection(self) -> None:
        frame, text_labels = self.collision_fixture()
        groups = np.repeat(np.arange(8), 20)
        attack = text_labels == "attack"
        encoder = RiskWeightedComparisonEncoder(1, random_state=2, max_swaps=0).fit(
            frame, attack, groups=groups
        )

        self.assertTrue(encoder.group_disjoint_selection_)
        self.assertEqual(encoder.transform(frame).dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
