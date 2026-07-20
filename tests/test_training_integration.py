from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.demo import generate_demo
from bitguard_bnn.export import export_run
from bitguard_bnn.trainer import run_training


class TrainingIntegrationTest(unittest.TestCase):
    def test_cost_aware_bnn_train_checkpoint_metrics_and_pruned_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "demo.csv"
            runs_path = root / "runs"
            generate_demo(data_path, rows=1_000, seed=29)
            config = copy.deepcopy(DEFAULTS)
            config["experiment"].update(
                {"name": "integration", "output_dir": str(runs_path), "seed": 29}
            )
            config["dataset"].update(
                {
                    "type": "csv",
                    "path": str(data_path),
                    "label_column": "behavior_label",
                    "raw_attack_column": "raw_attack",
                    "device_column": "device_id",
                    "time_column": "timestamp",
                }
            )
            config["split"].update(
                {
                    "strategy": "attack",
                    "held_out_attacks": ["novel_lowrate"],
                    "seed": 29,
                }
            )
            config["preprocess"].update(
                {
                    "feature_budget": 8,
                    "selection": "cost_aware",
                    "encoder": "sign",
                }
            )
            config["model"].update(
                {
                    "type": "cost_aware_bnn",
                    "hidden_dims": [8],
                    "dropout": 0.0,
                }
            )
            config["loss"].update({"lambda_feature": 0.01, "type": "weighted_ce"})
            config["training"].update(
                {
                    "epochs": 1,
                    "batch_size": 256,
                    "patience": 2,
                    "num_workers": 1,
                    "device": "cpu",
                    "amp": False,
                }
            )
            config["cascade"].update(
                {
                    "enabled": True,
                    "tiny_feature_budget": 4,
                    "hidden_dims": [4],
                    "threshold_grid_size": 21,
                }
            )
            config["temporal"]["enabled"] = False
            config["evaluation"].update(
                {"save_predictions": True, "make_plots": False, "benchmark_warmup": 1, "benchmark_repeats": 2}
            )
            config_path = root / "integration.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            run_dir = run_training(config_path)
            self.assertTrue((run_dir / "best_model.pt").exists())
            self.assertTrue((run_dir / "last_training_state.pt").exists())
            self.assertTrue((run_dir / "training_history.partial.csv").exists())
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metrics["classification"]["fixed_fpr"]["threshold_source"],
                "validation",
            )
            self.assertEqual(
                metrics["classification"]["fixed_fpr"]["score_pipeline"],
                "cascade",
            )
            self.assertIsNotNone(metrics["cascade"])

            import torch

            checkpoint_path = run_dir / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            gate_logits = checkpoint["state_dict"]["feature_gate.logits"]
            gate_logits[0] = -3.0
            checkpoint["state_dict"]["feature_gate.logits"] = gate_logits
            torch.save(checkpoint, checkpoint_path)

            export_dir = root / "edge"
            result = export_run(run_dir, export_dir)
            self.assertTrue(result["end_to_end_logit_parity_passed"])
            manifest = json.loads(
                (export_dir / "bitguard_edge_manifest.json").read_text(encoding="utf-8")
            )
            with np.load(export_dir / "bitguard_edge_weights.npz") as arrays:
                active_encoded_count = len(arrays["active_encoded_indices"])
                self.assertLess(active_encoded_count, int(checkpoint["input_dim"]))
                self.assertEqual(
                    manifest["layers"][0]["input_dimension"],
                    active_encoded_count,
                )
            self.assertTrue(manifest["end_to_end_logit_parity_passed"])
            self.assertTrue(manifest["preprocessing"]["open_set_requires_all_selected_features"])


if __name__ == "__main__":
    unittest.main()
