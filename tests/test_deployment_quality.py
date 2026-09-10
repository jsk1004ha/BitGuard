from __future__ import annotations

import copy
import unittest

from bitguard_bnn.deployment_quality import assess_deployment_candidate


class DeploymentQualityAssessmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "min_macro_f1": 0.88,
            "min_balanced_accuracy": 0.90,
            "required_classes": ["benign", "scan_like", "flood_like"],
            "min_required_class_support": 10,
            "min_required_class_recall": 0.85,
            "high_risk_classes": ["scan_like", "flood_like"],
            "min_high_risk_recall": 0.95,
            "fixed_fpr_target": 0.001,
            "min_attack_recall_at_fixed_fpr": 0.90,
            "max_observed_benign_fpr": 0.001,
            "max_ece": 0.03,
            "min_active_groups": 2,
        }
        self.metrics = {
            "macro_f1": 0.88,
            "balanced_accuracy": 0.90,
            "per_class": {
                "benign": {"recall": 0.85, "support": 10},
                "scan_like": {"recall": 0.95, "support": 10},
                "flood_like": {"recall": 0.95, "support": 10},
            },
            "fixed_fpr": {
                "attack_recall_at_benign_fpr_0.001": 0.90,
                "observed_benign_fpr_at_target_0.001": 0.001,
            },
            "expected_calibration_error_10_bin": 0.03,
        }
        self.model_summary = {"feature_gate_enabled": True, "active_groups": 2}

    def assess(
        self,
        metrics: dict[str, object] | None = None,
        model_summary: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return assess_deployment_candidate(
            self.metrics if metrics is None else metrics,
            self.model_summary if model_summary is None else model_summary,
            self.policy,
        )

    def assert_failed(self, check: str, assessment: dict[str, object]) -> None:
        self.assertFalse(assessment["eligible"])
        self.assertIn(check, assessment["failed_checks"])

    def test_exact_policy_boundaries_are_deployment_eligible(self) -> None:
        assessment = self.assess()

        self.assertTrue(assessment["eligible"])
        self.assertEqual(assessment["failed_checks"], [])

    def test_macro_f1_below_policy_fails(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["macro_f1"] = 0.879

        self.assert_failed("macro_f1", self.assess(metrics))

    def test_balanced_accuracy_below_policy_fails(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["balanced_accuracy"] = 0.899

        self.assert_failed("balanced_accuracy", self.assess(metrics))

    def test_minimum_probability_metrics_must_be_in_unit_interval(self) -> None:
        cases = (
            ("macro_f1", "macro_f1"),
            ("balanced_accuracy", "balanced_accuracy"),
        )
        for metric_name, check_name in cases:
            for actual in (-0.01, 1.01):
                with self.subTest(metric=metric_name, actual=actual):
                    metrics = copy.deepcopy(self.metrics)
                    metrics[metric_name] = actual

                    assessment = self.assess(metrics)

                    self.assert_failed(check_name, assessment)
                    self.assertEqual(
                        assessment["checks"][check_name]["actual"],  # type: ignore[index]
                        actual,
                    )

    def test_every_required_class_must_have_minimum_support(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["per_class"]["scan_like"]["support"] = 9  # type: ignore[index]

        self.assert_failed("required_class_support", self.assess(metrics))

    def test_every_required_class_must_meet_recall(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["per_class"]["benign"]["recall"] = 0.849  # type: ignore[index]

        self.assert_failed("required_class_recall", self.assess(metrics))

    def test_required_and_high_risk_recalls_must_be_in_unit_interval(self) -> None:
        for actual in (-0.01, 1.01):
            with self.subTest(actual=actual):
                metrics = copy.deepcopy(self.metrics)
                metrics["per_class"]["scan_like"]["recall"] = actual  # type: ignore[index]

                assessment = self.assess(metrics)

                self.assert_failed("required_class_recall", assessment)
                self.assertIn("high_risk_recall", assessment["failed_checks"])

    def test_required_class_support_must_be_finite_non_negative_integer(self) -> None:
        for actual in (-1, 10.5, float("inf")):
            with self.subTest(actual=actual):
                metrics = copy.deepcopy(self.metrics)
                metrics["per_class"]["scan_like"]["support"] = actual  # type: ignore[index]

                self.assert_failed("required_class_support", self.assess(metrics))

    def test_missing_required_class_fails_support_and_recall_closed(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        del metrics["per_class"]["scan_like"]  # type: ignore[index]

        assessment = self.assess(metrics)

        self.assert_failed("required_class_support", assessment)
        self.assertIn("required_class_recall", assessment["failed_checks"])

    def test_each_high_risk_class_must_meet_stricter_recall(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["per_class"]["flood_like"]["recall"] = 0.949  # type: ignore[index]

        self.assert_failed("high_risk_recall", self.assess(metrics))

    def test_fixed_fpr_attack_recall_below_policy_fails(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["fixed_fpr"]["attack_recall_at_benign_fpr_0.001"] = 0.899  # type: ignore[index]

        self.assert_failed("fixed_fpr_attack_recall", self.assess(metrics))

    def test_fixed_fpr_attack_recall_must_be_in_unit_interval(self) -> None:
        for actual in (-0.01, 1.01):
            with self.subTest(actual=actual):
                metrics = copy.deepcopy(self.metrics)
                metrics["fixed_fpr"]["attack_recall_at_benign_fpr_0.001"] = actual  # type: ignore[index]

                self.assert_failed("fixed_fpr_attack_recall", self.assess(metrics))

    def test_fixed_fpr_lookup_uses_the_metrics_suffix_contract(self) -> None:
        target = 0.000333333333
        policy = copy.deepcopy(self.policy)
        policy["fixed_fpr_target"] = target
        metrics = copy.deepcopy(self.metrics)
        metrics["fixed_fpr"] = {
            f"attack_recall_at_benign_fpr_{target:g}": 0.90,
            f"observed_benign_fpr_at_target_{target:g}": target,
        }
        policy["max_observed_benign_fpr"] = target

        assessment = assess_deployment_candidate(
            metrics,
            self.model_summary,
            policy,
        )

        self.assertTrue(assessment["eligible"])

    def test_observed_benign_fpr_above_policy_fails(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["fixed_fpr"]["observed_benign_fpr_at_target_0.001"] = 0.0011  # type: ignore[index]

        self.assert_failed("fixed_fpr_benign_fpr", self.assess(metrics))

    def test_maximum_probability_metrics_must_be_in_unit_interval(self) -> None:
        cases = (
            (
                "fixed_fpr",
                "observed_benign_fpr_at_target_0.001",
                "fixed_fpr_benign_fpr",
            ),
            (None, "expected_calibration_error_10_bin", "ece"),
        )
        for container_name, metric_name, check_name in cases:
            for actual in (-0.01, 1.01):
                with self.subTest(metric=metric_name, actual=actual):
                    metrics = copy.deepcopy(self.metrics)
                    container = (
                        metrics
                        if container_name is None
                        else metrics[container_name]
                    )
                    container[metric_name] = actual  # type: ignore[index]

                    assessment = self.assess(metrics)

                    self.assert_failed(check_name, assessment)
                    self.assertEqual(
                        assessment["checks"][check_name]["actual"],  # type: ignore[index]
                        actual,
                    )

    def test_ece_above_policy_fails(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics["expected_calibration_error_10_bin"] = 0.031

        self.assert_failed("ece", self.assess(metrics))

    def test_too_few_active_feature_groups_fails(self) -> None:
        model_summary = copy.deepcopy(self.model_summary)
        model_summary["active_groups"] = 1

        self.assert_failed("active_groups", self.assess(model_summary=model_summary))

    def test_active_groups_must_be_a_non_negative_integer(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["min_active_groups"] = 0
        for actual in (-1, 2.5):
            with self.subTest(actual=actual):
                model_summary = copy.deepcopy(self.model_summary)
                model_summary["active_groups"] = actual

                assessment = assess_deployment_candidate(
                    self.metrics,
                    model_summary,
                    policy,
                )

                self.assert_failed("active_groups", assessment)

    def test_missing_metrics_fail_closed_instead_of_being_skipped(self) -> None:
        cases = {
            "macro_f1": ("macro_f1",),
            "balanced_accuracy": ("balanced_accuracy",),
            "per_class": (
                "required_class_support",
                "required_class_recall",
                "high_risk_recall",
            ),
            "fixed_fpr": ("fixed_fpr_attack_recall", "fixed_fpr_benign_fpr"),
            "expected_calibration_error_10_bin": ("ece",),
        }
        for missing_metric, expected_failures in cases.items():
            with self.subTest(missing_metric=missing_metric):
                metrics = copy.deepcopy(self.metrics)
                del metrics[missing_metric]

                assessment = self.assess(metrics)

                self.assertFalse(assessment["eligible"])
                for expected_failure in expected_failures:
                    self.assertIn(expected_failure, assessment["failed_checks"])

    def test_missing_active_group_summary_fails_closed(self) -> None:
        self.assert_failed("active_groups", self.assess(model_summary={}))


if __name__ == "__main__":
    unittest.main()
