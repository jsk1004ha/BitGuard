from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bitguard_bnn.trainer import run_training  # noqa: E402


CLASSICAL = {"logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost"}


def _absolutize_paths(config: dict, config_path: Path) -> None:
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    for section, key in (
        ("dataset", "path"),
        ("dataset", "cross_path"),
        ("preprocess", "feature_cost_csv"),
        ("experiment", "output_dir"),
    ):
        value = config.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            config[section][key] = str(project_root / value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible BitGuard experiment matrix")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "fp32_mlp", "vanilla_bnn", "cost_aware_bnn"],
    )
    parser.add_argument("--encoders", nargs="+", default=["sign", "thermometer", "hybrid"])
    parser.add_argument("--losses", nargs="+", default=["weighted_ce", "focal"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2309, 2310, 2311])
    parser.add_argument("--output", type=Path, default=Path("results/experiment_matrix.csv"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    _absolutize_paths(base, config_path)
    records: list[dict[str, object]] = []
    for model_type in args.models:
        for encoder in args.encoders:
            for loss_type in args.losses:
                for seed in args.seeds:
                    config = copy.deepcopy(base)
                    config.setdefault("model", {})["type"] = model_type
                    config.setdefault("preprocess", {})["encoder"] = encoder
                    config.setdefault("loss", {})["type"] = loss_type
                    config.setdefault("experiment", {})["seed"] = seed
                    config.setdefault("split", {})["seed"] = seed
                    name = config["experiment"].get("name", "bitguard")
                    config["experiment"]["name"] = f"{name}_{model_type}_{encoder}_{loss_type}_s{seed}"
                    if model_type in CLASSICAL:
                        config.setdefault("cascade", {})["enabled"] = False
                    with tempfile.NamedTemporaryFile(
                        "w", suffix=".yaml", encoding="utf-8", delete=False
                    ) as handle:
                        yaml.safe_dump(config, handle, sort_keys=False)
                        temporary = Path(handle.name)
                    try:
                        run_dir = run_training(temporary)
                        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                        classification = metrics["classification"]
                        records.append(
                            {
                                "model": model_type,
                                "encoder": encoder,
                                "loss": loss_type,
                                "seed": seed,
                                "run_dir": str(run_dir),
                                "macro_f1": classification["macro_f1"],
                                "macro_auprc": classification["macro_auprc"],
                                "high_risk_false_negative_rate": classification[
                                    "high_risk_false_negative_rate"
                                ],
                            }
                        )
                    finally:
                        temporary.unlink(missing_ok=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(json.dumps({"output": str(output), "runs": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

