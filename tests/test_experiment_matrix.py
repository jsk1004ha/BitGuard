from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from scripts.run_experiment_matrix import experiment_combinations, main, summarize_results


class ExperimentMatrixTest(unittest.TestCase):
    def test_classical_models_ignore_neural_loss_variants(self) -> None:
        combinations = experiment_combinations(
            models=["logistic_regression", "vanilla_bnn"],
            encoders=["sign"],
            losses=["weighted_ce", "focal"],
            seeds=[11, 12],
        )

        classical = [row for row in combinations if row["model"] == "logistic_regression"]
        neural = [row for row in combinations if row["model"] == "vanilla_bnn"]

        self.assertEqual(len(classical), 2)
        self.assertEqual({row["loss"] for row in classical}, {"not_applicable"})
        self.assertEqual(len(neural), 4)
        self.assertEqual({row["loss"] for row in neural}, {"weighted_ce", "focal"})
        self.assertEqual({row["seed"] for row in combinations}, {11, 12})

    def test_summary_reports_sample_std_and_normal_95_percent_ci(self) -> None:
        records = pd.DataFrame(
            [
                {
                    "model": "vanilla_bnn",
                    "encoder": "sign",
                    "loss": "weighted_ce",
                    "seed": 11,
                    "macro_f1": 0.7,
                    "macro_auprc": 0.8,
                    "high_risk_false_negative_rate": 0.2,
                },
                {
                    "model": "vanilla_bnn",
                    "encoder": "sign",
                    "loss": "weighted_ce",
                    "seed": 12,
                    "macro_f1": 0.9,
                    "macro_auprc": 0.6,
                    "high_risk_false_negative_rate": 0.4,
                },
            ]
        )

        summary = summarize_results(records)

        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["run_count"], 2)
        self.assertAlmostEqual(row["macro_f1_mean"], 0.8)
        self.assertAlmostEqual(row["macro_f1_std"], 0.02**0.5)
        margin = 1.96 * (0.02**0.5) / (2**0.5)
        self.assertAlmostEqual(row["macro_f1_ci95_low"], 0.8 - margin)
        self.assertAlmostEqual(row["macro_f1_ci95_high"], 0.8 + margin)

    def test_main_writes_raw_and_summary_csvs_for_the_deduplicated_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {},
                        "preprocess": {},
                        "model": {},
                        "loss": {"type": "weighted_ce"},
                        "cascade": {},
                        "split": {},
                        "experiment": {"name": "matrix"},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "matrix.csv"
            submitted_configs: list[dict] = []

            def fake_run_training(temporary_config: Path) -> Path:
                submitted_configs.append(
                    yaml.safe_load(temporary_config.read_text(encoding="utf-8"))
                )
                run_dir = root / f"run-{len(submitted_configs)}"
                run_dir.mkdir()
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "classification": {
                                "macro_f1": 0.8,
                                "macro_auprc": 0.7,
                                "high_risk_false_negative_rate": 0.1,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return run_dir

            arguments = [
                "run_experiment_matrix.py",
                "--config",
                str(config_path),
                "--models",
                "logistic_regression",
                "vanilla_bnn",
                "--encoders",
                "sign",
                "--losses",
                "weighted_ce",
                "focal",
                "--seeds",
                "11",
                "12",
                "--output",
                str(output),
            ]
            with (
                patch("scripts.run_experiment_matrix.run_training", side_effect=fake_run_training),
                patch.object(sys, "argv", arguments),
                redirect_stdout(io.StringIO()),
            ):
                main()

            raw = pd.read_csv(output)
            summary = pd.read_csv(root / "matrix_summary.csv")
            self.assertEqual(len(submitted_configs), 6)
            self.assertEqual(len(raw), 6)
            self.assertEqual(len(summary), 3)
            self.assertEqual(set(summary["run_count"]), {2})
            classical = raw[raw["model"] == "logistic_regression"]
            self.assertEqual(set(classical["loss"]), {"not_applicable"})


if __name__ == "__main__":
    unittest.main()
