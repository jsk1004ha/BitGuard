from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "bitguard", "output_dir": "runs", "seed": 2309},
    "dataset": {
        "type": "csv",
        "chunk_size": 200_000,
        "max_rows_per_file": None,
        "max_rows_per_class": None,
        "drop_columns": [],
    },
    "split": {
        "strategy": "random",
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "held_out_devices": [],
        "held_out_attacks": [],
        "block_size": 10_000,
        "seed": 2309,
    },
    "preprocess": {
        "feature_budget": None,
        "selection": "f_score",
        "scaler": "robust",
        "encoder": "sign",
        "thermometer_bits": 2,
        "feature_cost_csv": None,
        "expert_features": [],
        "open_set": {
            "enabled": True,
            "confidence_threshold": 0.60,
            "benign_distance_quantile": 0.99,
            "max_known_false_unknown_rate": 0.02,
        },
    },
    "model": {
        "type": "vanilla_bnn",
        "hidden_dims": [64, 32],
        "dropout": 0.0,
        "binary_first_layer": True,
        "gate_temperature": 1.0,
    },
    "loss": {
        "type": "weighted_ce",
        "focal_gamma": 2.0,
        "class_weighted": True,
        "lambda_feature": 0.0,
        "beta_fn": 0.0,
        "gamma_fp": 0.0,
        "distillation_alpha": 0.0,
        "distillation_temperature": 2.0,
    },
    "training": {
        "epochs": 30,
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 6,
        "num_workers": 0,
        "device": "auto",
        "amp": False,
        "gradient_clip": 5.0,
        "selection_weights": {"macro_f1": 0.5, "macro_auprc": 0.3, "attack_recall": 0.2},
    },
    "cascade": {
        "enabled": False,
        "boolean_fast_path_enabled": True,
        "boolean_fast_path_features": [
            "packet_rate",
            "burst_score",
            "syn_ratio",
            "unique_destination_ip_ratio",
            "failed_connection_score",
        ],
        "tiny_feature_budget": 8,
        "hidden_dims": [16],
        "min_attack_recall": 0.995,
        "threshold_grid_size": 201,
        "temporal_penalty": 0.30,
        "use_temporal_state": True,
        "false_negative_cost": 0.10,
        "device_criticality_default": 0.0,
        "device_criticality": {},
    },
    "temporal": {
        "enabled": False,
        "increment": 2,
        "decay": 1,
        "evidence_threshold": 0.45,
        "max_devices": 4096,
        "risk_weights": {
            "model": 0.55,
            "scan": 0.13,
            "flood": 0.13,
            "beacon": 0.10,
            "unknown": 0.14,
            "benign": 0.08,
        },
        "action_thresholds": [0.18, 0.30, 0.44, 0.60, 0.78],
    },
    "evaluation": {
        "high_risk_labels": ["scan_like", "flood_like", "exfil_like", "unknown_like"],
        "save_predictions": True,
        "make_plots": True,
        "benchmark_warmup": 20,
        "benchmark_repeats": 100,
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULTS, raw)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root.resolve())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    fractions = [
        float(config["split"]["train_fraction"]),
        float(config["split"]["validation_fraction"]),
        float(config["split"]["test_fraction"]),
    ]
    if any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must be positive and sum to 1.0")
    if config["split"]["strategy"] not in {
        "random",
        "device",
        "attack",
        "time",
        "sequence",
        "block",
        "cross",
    }:
        raise ValueError("unsupported split.strategy")
    if config["preprocess"]["encoder"] not in {"none", "sign", "thermometer", "hybrid"}:
        raise ValueError("preprocess.encoder must be none/sign/thermometer/hybrid")
    bits = int(config["preprocess"]["thermometer_bits"])
    if not 2 <= bits <= 4:
        raise ValueError("thermometer_bits must be between 2 and 4")
    if config["model"]["type"] not in {
        "fp32_mlp",
        "vanilla_bnn",
        "cost_aware_bnn",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
    }:
        raise ValueError("unsupported model.type")
    thresholds = config["temporal"]["action_thresholds"]
    if len(thresholds) != 5 or list(thresholds) != sorted(thresholds):
        raise ValueError("temporal.action_thresholds must contain five increasing values")
    selection_weights = config["training"].get("selection_weights", {})
    if not np.isclose(sum(float(value) for value in selection_weights.values()), 1.0):
        raise ValueError("training.selection_weights must sum to 1.0")
    for device, risk in config["cascade"].get("device_criticality", {}).items():
        if not 0.0 <= float(risk) <= 1.0:
            raise ValueError(f"device criticality for {device!r} must be in [0, 1]")
    if int(config["temporal"].get("max_devices", 4096)) <= 0:
        raise ValueError("temporal.max_devices must be positive")


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def create_run_dir(config: dict[str, Any]) -> Path:
    root = resolve_path(config, config["experiment"]["output_dir"])
    assert root is not None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / str(config["experiment"]["name"]) / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = root / str(config["experiment"]["name"]) / f"{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def save_yaml(data: dict[str, Any], path: Path) -> None:
    clean = {key: value for key, value in data.items() if not key.startswith("_")}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False, allow_unicode=True)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def environment_manifest() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = None
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": commit,
    }
    try:
        import torch

        manifest.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except ImportError:
        manifest["torch"] = None
    return manifest
