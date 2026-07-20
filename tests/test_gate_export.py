from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.models import build_model, classifier_active_inputs, feature_gate_summary


class GateExportTest(unittest.TestCase):
    def test_gate_selection_maps_groups_to_encoded_inputs_with_logit_parity(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["model"].update(
            {
                "type": "cost_aware_bnn",
                "hidden_dims": [5],
                "dropout": 0.0,
                "binary_first_layer": True,
            }
        )
        groups = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
        model = build_model(
            config,
            input_dim=5,
            output_dim=2,
            input_groups=groups,
            feature_costs=np.ones(3, dtype=np.float32),
        )
        assert model.feature_gate is not None
        with torch.no_grad():
            model.feature_gate.logits.copy_(torch.tensor([3.0, -3.0, 2.0]))
        active_groups, active_inputs = classifier_active_inputs(model)
        self.assertEqual(active_groups.tolist(), [0, 2])
        self.assertEqual(active_inputs.tolist(), [0, 1, 3, 4])
        summary = feature_gate_summary(model)
        self.assertEqual(summary["active_groups"], 2)
        self.assertEqual(summary["active_encoded_inputs"], 4)

        model.eval()
        values = torch.randn(7, 5)
        with torch.inference_mode():
            gated = model.feature_gate(values)
            full = model.blocks[0][0](gated)
            linear = model.blocks[0][0]
            binary_weight = torch.where(linear.weight >= 0, 1.0, -1.0)
            pruned = values[:, active_inputs] @ binary_weight[:, active_inputs].T
        self.assertTrue(torch.allclose(full, pruned, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
