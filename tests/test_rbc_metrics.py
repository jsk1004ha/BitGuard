from __future__ import annotations

import unittest

import numpy as np

from bitguard_bnn.rbc_metrics import (
    binomial_fpr_upper_bound,
    calibrate_benign_threshold,
    evaluate_rbc,
)


class BenignThresholdCalibrationTest(unittest.TestCase):
    def test_empirical_calibration_uses_strict_greater_than_rule(self) -> None:
        result = calibrate_benign_threshold(
            [0.1, 0.1, 0.2, 0.3], 0.25, method="empirical"
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["threshold_rule"], "score > threshold")
        self.assertEqual(result["threshold"], 0.2)
        self.assertEqual(result["empirical_false_positives"], 1)
        self.assertEqual(result["empirical_fpr"], 0.25)

    def test_np_calibration_reports_insufficient_sample_size(self) -> None:
        result = calibrate_benign_threshold(
            np.linspace(0.0, 1.0, 100),
            0.001,
            method="np_order_statistic",
            confidence=0.95,
        )

        self.assertEqual(result["status"], "insufficient_n")
        self.assertEqual(result["minimum_n_benign"], 2995)
        self.assertIsNone(result["threshold"])

    def test_np_calibration_returns_valid_order_statistic(self) -> None:
        scores = np.arange(3000, dtype=np.float64)
        result = calibrate_benign_threshold(
            scores, 0.001, method="np_order_statistic", confidence=0.95
        )

        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(result["violation_probability"], 0.05)
        self.assertEqual(result["threshold"], scores[-1])
        self.assertEqual(result["empirical_false_positives"], 0)


class RbcEvaluationTest(unittest.TestCase):
    def test_fixed_threshold_type_unknown_group_and_benign_metrics(self) -> None:
        result = evaluate_rbc(
            scores=[0.5, 0.51, 0.9, 0.2, 0.8, 0.7],
            true_labels=["benign", "benign", "scan", "scan", "flood", "novel"],
            threshold=0.5,
            known_attack_labels=["scan", "flood", "unused_known"],
            unknown_attack_labels=["novel"],
            type_predictions=["unknown", "unknown", "scan", "scan", "scan", "unknown"],
            groups={"device": ["a", "a", "a", "a", "b", "b"]},
            test_iid=True,
        )

        self.assertEqual(result["detection"]["tpr"], 0.75)
        self.assertEqual(result["detection"]["fnr"], 0.25)
        self.assertEqual(result["benign_only"]["fpr"], 0.5)
        self.assertEqual(result["attack_type"]["conditional_on_detection"]["labels"], ["scan", "flood", "unused_known"])
        self.assertEqual(result["attack_type"]["end_to_end"]["per_class"]["scan"]["support"], 2)
        self.assertEqual(result["attack_type"]["end_to_end"]["per_class"]["scan"]["recall"], 0.5)
        self.assertEqual(result["unknown_attack"]["detection_rate"], 1.0)
        self.assertEqual(result["unknown_attack"]["rejection_rate"], 1.0)
        self.assertEqual(result["unknown_attack"]["benign_to_unknown_rate"], 0.5)
        self.assertEqual(result["grouped_detection"]["device"]["groups"]["a"]["support"], 2)
        self.assertEqual(result["grouped_detection"]["device"]["worst_tpr"], 0.5)
        self.assertEqual(result["test_fpr_upper_bound"]["status"], "ok")

    def test_tied_score_at_threshold_is_not_an_attack(self) -> None:
        result = evaluate_rbc(
            [0.5, 0.5],
            ["benign", "attack"],
            0.5,
            known_attack_labels=["attack"],
        )

        self.assertEqual(result["benign_only"]["false_positives"], 0)
        self.assertEqual(result["detection"]["tpr"], 0.0)

    def test_test_bound_requires_iid_flag(self) -> None:
        result = binomial_fpr_upper_bound(0, 2995, confidence=0.95, iid=False)

        self.assertEqual(result["status"], "not_applicable_non_iid")
        self.assertIsNone(result["upper_bound"])

    def test_zero_false_positive_bound_matches_closed_form(self) -> None:
        result = binomial_fpr_upper_bound(0, 2995, confidence=0.95, iid=True)

        expected = 1.0 - 0.05 ** (1.0 / 2995)
        self.assertAlmostEqual(result["upper_bound"], expected)


if __name__ == "__main__":
    unittest.main()
