from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bitguard_bnn.trainer import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[115, 64, 32, 16, 8])
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    for section, key in (
        ("dataset", "path"),
        ("dataset", "cross_path"),
        ("preprocess", "feature_cost_csv"),
        ("experiment", "output_dir"),
    ):
        value = base.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            base[section][key] = str(project_root / value)
    results: list[dict[str, str | int]] = []
    for budget in args.budgets:
        config = copy.deepcopy(base)
        config.setdefault("preprocess", {})["feature_budget"] = budget
        config.setdefault("experiment", {})["name"] = f"{config['experiment'].get('name', 'bitguard')}_f{budget}"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
            temporary = Path(handle.name)
        try:
            run_dir = run_training(temporary)
            results.append({"budget": budget, "run_dir": str(run_dir)})
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
