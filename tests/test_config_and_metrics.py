from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from bitguard_bnn.config import DEFAULTS, load_config, validate_config
from bitguard_bnn.metrics import calibrate_fixed_fpr_thresholds, classification_metrics
from bitguard_bnn.trainer import neural_validation_metrics


ROOT = Path(__file__).resolve().parents[1]


class SplitConfigurationTest(unittest.TestCase):
    def test_neural_models_require_at_least_one_hidden_layer(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["model"]["hidden_dims"] = []

        with self.assertRaisesRegex(ValueError, "hidden_dims"):
            validate_config(config)

    def test_held_out_attacks_require_attack_split(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["split"].update(
            {"strategy": "time", "held_out_attacks": ["data_exfiltration"]}
        )

        with self.assertRaisesRegex(ValueError, "held_out_attacks"):
            validate_config(config)

    def test_held_out_devices_require_device_split(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["split"].update(
            {"strategy": "attack", "held_out_devices": ["device-1"]}
        )

        with self.assertRaisesRegex(ValueError, "held_out_devices"):
            validate_config(config)

    def test_dataset_configs_do_not_mix_split_protocols(self) -> None:
        nbaiot = load_config(ROOT / "configs" / "nbaiot.yaml")
        botiot = load_config(ROOT / "configs" / "botiot.yaml")
        nbaiot_attack = load_config(ROOT / "configs" / "nbaiot_attack.yaml")
        botiot_attack = load_config(ROOT / "configs" / "botiot_attack.yaml")

        self.assertEqual(nbaiot["split"]["strategy"], "device")
        self.assertEqual(nbaiot["split"]["held_out_attacks"], [])
        self.assertEqual(botiot["split"]["strategy"], "time")
        self.assertEqual(botiot["split"]["held_out_attacks"], [])
        self.assertEqual(nbaiot_attack["split"]["strategy"], "attack")
        self.assertEqual(nbaiot_attack["split"]["held_out_attacks"], ["mirai_scan"])
        self.assertEqual(botiot_attack["split"]["strategy"], "attack")
        self.assertEqual(
            botiot_attack["split"]["held_out_attacks"],
            ["keylogging", "data_exfiltration"],
        )

    def test_gate_schedule_and_minimum_fraction_are_validated(self) -> None:
        cases = (
            ("model", "minimum_active_fraction", -0.01),
            ("model", "minimum_active_fraction", 1.01),
            ("loss", "feature_penalty_warmup_fraction", -0.01),
            ("loss", "feature_penalty_warmup_fraction", 1.01),
            ("loss", "feature_penalty_ramp_fraction", -0.01),
            ("loss", "feature_penalty_ramp_fraction", 1.01),
        )
        for section, name, value in cases:
            with self.subTest(section=section, name=name, value=value):
                config = copy.deepcopy(DEFAULTS)
                config[section][name] = value
                with self.assertRaisesRegex(ValueError, name):
                    validate_config(config)

    def test_repository_full_profiles_enable_deployment_candidate_policy(self) -> None:
        for filename in ("botiot.yaml", "nbaiot.yaml"):
            with self.subTest(filename=filename):
                config = load_config(ROOT / "configs" / "full" / filename)
                self.assertGreater(config["model"]["minimum_active_fraction"], 0.0)
                self.assertGreater(
                    config["loss"]["feature_penalty_warmup_fraction"], 0.0
                )
                self.assertGreater(
                    config["loss"]["feature_penalty_ramp_fraction"], 0.0
                )
                gate = config["evaluation"]["deployment_candidate"]
                self.assertTrue(gate["enabled"])
                self.assertGreaterEqual(gate["min_balanced_accuracy"], 0.90)
                self.assertGreaterEqual(gate["min_macro_f1"], 0.88)
                self.assertGreaterEqual(gate["min_required_class_recall"], 0.85)
                self.assertGreaterEqual(gate["min_high_risk_recall"], 0.95)
                self.assertLessEqual(gate["max_observed_benign_fpr"], 0.001)
                self.assertGreaterEqual(
                    gate["min_attack_recall_at_fixed_fpr"], 0.90
                )
                self.assertLessEqual(gate["max_ece"], 0.03)

    def test_enabled_deployment_policy_requires_explicit_risk_classes(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        policy = config["evaluation"]["deployment_candidate"]
        policy["enabled"] = True

        with self.assertRaisesRegex(ValueError, "required_classes"):
            validate_config(config)

        policy["required_classes"] = ["benign", "scan_like"]
        with self.assertRaisesRegex(ValueError, "high_risk_classes"):
            validate_config(config)

        policy["high_risk_classes"] = ["benign"]
        with self.assertRaisesRegex(ValueError, "high_risk_classes"):
            validate_config(config)

    def test_enabled_deployment_policy_requires_out_of_core_parquet_storage(
        self,
    ) -> None:
        config = copy.deepcopy(DEFAULTS)
        policy = config["evaluation"]["deployment_candidate"]
        policy.update(
            {
                "enabled": True,
                "required_classes": ["benign", "scan_like"],
                "high_risk_classes": ["scan_like"],
            }
        )

        with self.assertRaisesRegex(ValueError, "dataset.storage=parquet"):
            validate_config(config)

    def test_deployment_policy_rejects_unknown_fields(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["evaluation"]["deployment_candidate"]["minimum_precision"] = 0.9

        with self.assertRaisesRegex(ValueError, "unknown.*minimum_precision"):
            validate_config(config)

    def test_deployment_policy_rejects_missing_fields(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        del config["evaluation"]["deployment_candidate"]["max_ece"]

        with self.assertRaisesRegex(ValueError, "missing.*max_ece"):
            validate_config(config)


class NeuralValidationMetricsTest(unittest.TestCase):
    training_config = {
        "selection_weights": {
            "macro_f1": 0.5,
            "macro_auprc": 0.3,
            "attack_recall": 0.2,
        }
    }

    def test_attack_recall_is_worst_supported_attack_subtype_recall(self) -> None:
        y_validation = np.asarray([0, 1, 2], dtype=np.int64)
        validation_probability = np.asarray(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.05, 0.90],
                [0.05, 0.05, 0.90],
            ],
            dtype=np.float64,
        )

        metrics = neural_validation_metrics(
            y_validation,
            validation_probability,
            self.training_config,
        )

        self.assertEqual(metrics["validation_attack_recall"], 0.0)

    def test_attack_recall_ignores_attack_subtypes_without_support(self) -> None:
        y_validation = np.asarray([0, 1], dtype=np.int64)
        validation_probability = np.asarray(
            [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05]], dtype=np.float64
        )

        metrics = neural_validation_metrics(
            y_validation,
            validation_probability,
            self.training_config,
        )

        self.assertEqual(metrics["validation_attack_recall"], 1.0)

    def test_attack_recall_is_zero_without_supported_attack_subtypes(self) -> None:
        y_validation = np.asarray([0, 0], dtype=np.int64)
        validation_probability = np.asarray(
            [[0.90, 0.05, 0.05], [0.80, 0.10, 0.10]], dtype=np.float64
        )

        metrics = neural_validation_metrics(
            y_validation,
            validation_probability,
            self.training_config,
        )

        self.assertEqual(metrics["validation_attack_recall"], 0.0)


class FixedFPRCalibrationTest(unittest.TestCase):
    probability_labels = ["benign", "scan_like"]

    def test_thresholds_are_calibrated_on_validation_benign_scores(self) -> None:
        validation_labels = np.asarray(
            ["benign", "benign", "scan_like", "scan_like"]
        )
        validation_probabilities = np.asarray(
            [[0.99, 0.01], [0.90, 0.10], [0.30, 0.70], [0.10, 0.90]],
            dtype=np.float64,
        )

        thresholds = calibrate_fixed_fpr_thresholds(
            validation_labels,
            self.probability_labels,
            validation_probabilities,
            target_fprs=(0.5,),
        )

        self.assertEqual(set(thresholds), {0.5})
        self.assertAlmostEqual(thresholds[0.5], 0.10)

    def test_test_metrics_use_provided_thresholds_without_recalibration(self) -> None:
        y_true = np.asarray(["benign", "benign", "scan_like", "scan_like"])
        probabilities = np.asarray(
            [[0.95, 0.05], [0.85, 0.15], [0.40, 0.60], [0.20, 0.80]],
            dtype=np.float64,
        )
        y_pred = np.asarray(self.probability_labels)[probabilities.argmax(axis=1)]

        metrics = classification_metrics(
            y_true,
            y_pred,
            self.probability_labels,
            probabilities,
            ["scan_like"],
            operating_thresholds={0.5: 0.10},
        )

        fixed_fpr = metrics["fixed_fpr"]
        self.assertEqual(fixed_fpr["threshold_source"], "validation")
        self.assertAlmostEqual(fixed_fpr["threshold_at_benign_fpr_0.5"], 0.10)
        self.assertAlmostEqual(fixed_fpr["observed_benign_fpr_at_target_0.5"], 0.5)
        self.assertAlmostEqual(fixed_fpr["attack_recall_at_benign_fpr_0.5"], 1.0)

    def test_unattainable_small_fpr_uses_threshold_above_validation_maximum(self) -> None:
        labels = np.asarray(["benign", "benign", "benign", "scan_like"])
        probabilities = np.asarray(
            [[0.99, 0.01], [0.90, 0.10], [0.80, 0.20], [0.10, 0.90]],
            dtype=np.float64,
        )
        thresholds = calibrate_fixed_fpr_thresholds(
            labels,
            self.probability_labels,
            probabilities,
            target_fprs=(0.01,),
        )
        predicted = np.asarray(self.probability_labels)[probabilities.argmax(axis=1)]
        metrics = classification_metrics(
            labels,
            predicted,
            self.probability_labels,
            probabilities,
            ["scan_like"],
            operating_thresholds=thresholds,
        )
        self.assertLessEqual(
            metrics["fixed_fpr"]["observed_benign_fpr_at_target_0.01"],
            0.01,
        )

    def test_unattainable_small_fpr_is_preserved_for_float32_probabilities(self) -> None:
        labels = np.asarray(["benign", "benign", "benign", "scan_like"])
        probabilities = np.asarray(
            [[0.99, 0.01], [0.90, 0.10], [0.80, 0.20], [0.10, 0.90]],
            dtype=np.float32,
        )
        thresholds = calibrate_fixed_fpr_thresholds(
            labels,
            self.probability_labels,
            probabilities,
            target_fprs=(0.01,),
        )
        predicted = np.asarray(self.probability_labels)[probabilities.argmax(axis=1)]

        metrics = classification_metrics(
            labels,
            predicted,
            self.probability_labels,
            probabilities,
            ["scan_like"],
            operating_thresholds=thresholds,
        )

        self.assertEqual(
            metrics["fixed_fpr"]["observed_benign_fpr_at_target_0.01"],
            0.0,
        )

    def test_fixed_fpr_metrics_are_omitted_without_calibrated_thresholds(self) -> None:
        y_true = np.asarray(["benign", "scan_like"])
        probabilities = np.asarray([[0.90, 0.10], [0.20, 0.80]], dtype=np.float64)
        y_pred = np.asarray(self.probability_labels)[probabilities.argmax(axis=1)]

        metrics = classification_metrics(
            y_true,
            y_pred,
            self.probability_labels,
            probabilities,
            ["scan_like"],
        )

        self.assertEqual(metrics["fixed_fpr"], {})


if __name__ == "__main__":
    unittest.main()
