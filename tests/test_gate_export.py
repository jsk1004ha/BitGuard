from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bitguard_bnn.config import DEFAULTS, save_yaml
from bitguard_bnn.export import export_run
from bitguard_bnn.models import build_model, classifier_active_inputs, feature_gate_summary
from bitguard_bnn.preprocess import BinaryFeatureEncoder, FeaturePreprocessor, IdentityScaler


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

    def test_minimum_active_fraction_exports_projected_mask_and_columns(self) -> None:
        config = copy.deepcopy(DEFAULTS)
        config["model"].update(
            {
                "type": "cost_aware_bnn",
                "hidden_dims": [5],
                "dropout": 0.0,
                "binary_first_layer": True,
                "minimum_active_fraction": 2.0 / 3.0,
            }
        )
        config["preprocess"]["encoder"] = "sign"
        groups = np.arange(3, dtype=np.int64)
        model = build_model(
            config,
            input_dim=3,
            output_dim=2,
            input_groups=groups,
            feature_costs=np.ones(3, dtype=np.float32),
        )
        assert model.feature_gate is not None
        with torch.no_grad():
            model.feature_gate.logits.copy_(torch.tensor([-3.0, -1.0, -2.0]))
        model.eval()

        expected_groups, expected_inputs = classifier_active_inputs(model)
        expected_mask = model.feature_gate.hard_mask().detach().cpu().numpy().astype(np.uint8)
        self.assertEqual(expected_groups.tolist(), [1, 2])
        self.assertEqual(expected_inputs.tolist(), [1, 2])

        values = torch.randn(7, 3)
        with torch.inference_mode():
            gated = model.feature_gate(values)
            linear = model.blocks[0][0]
            binary_weight = torch.where(linear.weight >= 0, 1.0, -1.0)
            full = linear(gated)
            pruned = values[:, expected_inputs] @ binary_weight[:, expected_inputs].T
        self.assertTrue(torch.allclose(full, pruned, atol=1e-6, rtol=1e-6))

        preprocessor = FeaturePreprocessor(config)
        preprocessor.selected_features = ["first", "second", "third"]
        preprocessor.selected_indices = np.arange(3, dtype=np.int64)
        preprocessor.feature_costs = np.ones(3, dtype=np.float32)
        preprocessor.imputer.statistics_ = np.zeros(3, dtype=np.float64)
        preprocessor.scaler = IdentityScaler()
        preprocessor.encoder = BinaryFeatureEncoder(
            "sign", 2, thresholds=np.zeros(3, dtype=np.float32)
        )
        preprocessor.benign_center = np.zeros(3, dtype=np.float32)
        preprocessor.fitted = True

        checkpoint = {
            "state_dict": model.state_dict(),
            "model_type": "cost_aware_bnn",
            "input_dim": 3,
            "output_dim": 2,
            "output_labels": ["benign", "flood"],
            "hidden_dims": [5],
            "dropout": 0.0,
            "binary_first_layer": True,
            "input_groups": torch.from_numpy(groups),
            "feature_costs": torch.ones(3, dtype=torch.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            export_dir = root / "edge"
            run_dir.mkdir()
            save_yaml(config, run_dir / "resolved_config.yaml")
            preprocessor.save(run_dir / "preprocessor.joblib")
            torch.save(checkpoint, run_dir / "best_model.pt")

            result = export_run(run_dir, export_dir)
            self.assertTrue(result["end_to_end_logit_parity_passed"])
            manifest = json.loads(
                (export_dir / "bitguard_edge_manifest.json").read_text(encoding="utf-8")
            )
            with np.load(export_dir / "bitguard_edge_weights.npz") as arrays:
                np.testing.assert_array_equal(arrays["feature_gate_hard_mask"], expected_mask)
                np.testing.assert_array_equal(
                    arrays["active_original_feature_indices"], expected_groups
                )
                np.testing.assert_array_equal(arrays["active_encoded_indices"], expected_inputs)
                self.assertEqual(arrays["layer_0_input_bits"].tolist(), [2])
            self.assertEqual(manifest["layers"][0]["input_dimension"], 2)


if __name__ == "__main__":
    unittest.main()
