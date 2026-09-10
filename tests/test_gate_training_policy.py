from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from bitguard_bnn import losses
from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.models import build_model, classifier_active_inputs, feature_gate_summary


class FeaturePenaltyScheduleTest(unittest.TestCase):
    def _coefficient(
        self,
        loss_config: dict[str, float],
        optimizer_step: int,
        total_optimizer_steps: int,
    ) -> float:
        coefficient = getattr(losses, "feature_penalty_coefficient", None)
        self.assertIsNotNone(
            coefficient,
            "losses.feature_penalty_coefficient must define the optimizer-step schedule",
        )
        return float(coefficient(loss_config, optimizer_step, total_optimizer_steps))

    def test_warmup_and_ramp_boundaries_use_fractions_of_total_optimizer_steps(self) -> None:
        loss_config = {
            "lambda_feature": 0.4,
            "feature_penalty_warmup_fraction": 0.10,
            "feature_penalty_ramp_fraction": 0.20,
        }

        expected_by_step = {
            0: 0.0,
            9: 0.0,
            10: 0.0,
            20: 0.2,
            30: 0.4,
            31: 0.4,
            100: 0.4,
        }
        for optimizer_step, expected in expected_by_step.items():
            with self.subTest(optimizer_step=optimizer_step):
                self.assertAlmostEqual(
                    self._coefficient(loss_config, optimizer_step, 100),
                    expected,
                )

    def test_schedule_is_independent_of_rows_and_epoch_metadata(self) -> None:
        base = {
            "lambda_feature": 0.6,
            "feature_penalty_warmup_fraction": 0.10,
            "feature_penalty_ramp_fraction": 0.20,
        }
        with_training_metadata = {
            **base,
            "epoch": 99,
            "seen_rows": 10_000_000,
            "rows_per_epoch": 123,
        }

        self.assertAlmostEqual(self._coefficient(base, 20, 100), 0.3)
        self.assertAlmostEqual(
            self._coefficient(with_training_metadata, 20, 100),
            self._coefficient(base, 20, 100),
        )


class MinimumActiveFeatureGateTest(unittest.TestCase):
    def test_all_negative_logits_keep_top_fraction_consistent_across_gate_views(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["model"].update(
            {
                "type": "cost_aware_bnn",
                "hidden_dims": [4],
                "dropout": 0.0,
                "minimum_active_fraction": 0.5,
            }
        )
        input_groups = np.asarray([0, 0, 1, 2, 3, 3, 4], dtype=np.int64)
        model = build_model(
            config,
            input_dim=len(input_groups),
            output_dim=2,
            input_groups=input_groups,
            feature_costs=np.ones(5, dtype=np.float32),
        )
        gate = model.feature_gate
        self.assertIsNotNone(gate)
        assert gate is not None
        with torch.no_grad():
            gate.logits.copy_(torch.tensor([-5.0, -1.0, -4.0, -2.0, -3.0]))

        hard_mask_method = getattr(gate, "hard_mask", None)
        self.assertIsNotNone(
            hard_mask_method,
            "FeatureGate.hard_mask must be the shared minimum-selection policy",
        )
        hard_mask = hard_mask_method().detach().cpu()
        expected_group_mask = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        self.assertTrue(torch.equal(hard_mask, expected_group_mask))

        expected_groups = [1, 3, 4]
        expected_inputs = [2, 4, 5, 6]
        self.assertEqual(gate.selected_groups().detach().cpu().tolist(), expected_groups)
        active_groups, active_inputs = classifier_active_inputs(model)
        self.assertEqual(active_groups.tolist(), expected_groups)
        self.assertEqual(active_inputs.tolist(), expected_inputs)

        summary = feature_gate_summary(model)
        self.assertEqual(summary["active_groups"], 3)
        self.assertEqual(summary["active_encoded_inputs"], 4)
        self.assertEqual(summary["active_group_indices"], expected_groups)
        self.assertEqual(summary["minimum_active_fraction"], 0.5)
        self.assertEqual(summary["minimum_active_groups"], 3)

        values = torch.ones(1, len(input_groups))
        expected_encoded_mask = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]])
        self.assertTrue(torch.equal(gate(values).detach().cpu(), expected_encoded_mask))

        normalized_cost = gate.normalized_cost()
        self.assertAlmostEqual(float(normalized_cost.detach()), 3.0 / 5.0, places=6)
        normalized_cost.backward()
        self.assertIsNotNone(gate.logits.grad)


if __name__ == "__main__":
    unittest.main()
