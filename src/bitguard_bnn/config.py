from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "bitguard", "output_dir": "runs", "seed": 2309},
    "dataset": {
        "type": "csv",
        "storage": "csv",
        "shard_manifest": None,
        "chunk_size": 200_000,
        "record_batch_rows": 65_536,
        "shard_target_rows": 1_000_000,
        "quantile_sketch_capacity": 200_000,
        "max_rows_per_file": None,
        "max_rows_per_class": None,
        "max_loaded_rows": None,
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
        "minimum_active_fraction": 0.0,
    },
    "loss": {
        "type": "weighted_ce",
        "focal_gamma": 2.0,
        "class_weighted": True,
        "lambda_feature": 0.0,
        "feature_penalty_warmup_fraction": 0.0,
        "feature_penalty_ramp_fraction": 0.0,
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
        "checkpoint_every_epochs": 1,
        "checkpoint_every_steps": 1000,
        "shuffle_buffer_rows": 262144,
        "resume_from": None,
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
        "fixed_fpr_targets": [1e-2, 1e-3],
        "deployment_candidate": {
            "enabled": False,
            "required_classes": [],
            "min_required_class_support": 1,
            "min_required_class_recall": 0.85,
            "high_risk_classes": [],
            "min_high_risk_recall": 0.95,
            "min_balanced_accuracy": 0.90,
            "min_macro_f1": 0.88,
            "fixed_fpr_target": 1e-3,
            "min_attack_recall_at_fixed_fpr": 0.90,
            "max_observed_benign_fpr": 1e-3,
            "max_ece": 0.03,
            "min_active_groups": 1,
        },
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
    configs_root = next(
        (parent for parent in config_path.parents if parent.name == "configs"),
        None,
    )
    project_root = configs_root.parent if configs_root is not None else Path.cwd()
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root.resolve())
    validate_config(config)
    return config


def _validated_selection_weights(value: Any) -> dict[str, float]:
    expected = {"macro_f1", "macro_auprc", "attack_recall"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(
            "training.selection_weights must contain exactly macro_f1, "
            "macro_auprc, and attack_recall"
        )
    normalized: dict[str, float] = {}
    for name in expected:
        weight = value[name]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not math.isfinite(float(weight))
            or not 0.0 <= float(weight) <= 1.0
        ):
            raise ValueError("training.selection_weights values must be finite in [0, 1]")
        normalized[name] = float(weight)
    if not math.isclose(
        sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("training.selection_weights must sum to 1.0")
    return normalized


def _finite_unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite in [0, 1]")
    return float(value)


_DEPLOYMENT_CANDIDATE_POLICY_KEYS = frozenset(
    {
        "enabled",
        "required_classes",
        "min_required_class_support",
        "min_required_class_recall",
        "high_risk_classes",
        "min_high_risk_recall",
        "min_balanced_accuracy",
        "min_macro_f1",
        "fixed_fpr_target",
        "min_attack_recall_at_fixed_fpr",
        "max_observed_benign_fpr",
        "max_ece",
        "min_active_groups",
    }
)


def _validate_deployment_candidate_policy(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("evaluation.deployment_candidate must be a mapping")
    fields = set(value)
    missing = _DEPLOYMENT_CANDIDATE_POLICY_KEYS - fields
    unknown = fields - _DEPLOYMENT_CANDIDATE_POLICY_KEYS
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if unknown:
            details.append(
                f"unknown fields: {', '.join(sorted(str(name) for name in unknown))}"
            )
        raise ValueError(
            "evaluation.deployment_candidate has " + "; ".join(details)
        )
    if not isinstance(value.get("enabled"), bool):
        raise ValueError("evaluation.deployment_candidate.enabled must be boolean")
    for name in (
        "min_required_class_recall",
        "min_high_risk_recall",
        "min_balanced_accuracy",
        "min_macro_f1",
        "fixed_fpr_target",
        "min_attack_recall_at_fixed_fpr",
        "max_observed_benign_fpr",
        "max_ece",
    ):
        _finite_unit_interval(
            value.get(name), f"evaluation.deployment_candidate.{name}"
        )
    if float(value["fixed_fpr_target"]) <= 0.0:
        raise ValueError(
            "evaluation.deployment_candidate.fixed_fpr_target must be positive"
        )
    for name in ("required_classes", "high_risk_classes"):
        labels = value.get(name)
        if (
            not isinstance(labels, list)
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                f"evaluation.deployment_candidate.{name} must contain unique labels"
            )
    for name, minimum in (
        ("min_required_class_support", 1),
        ("min_active_groups", 0),
    ):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise ValueError(
                f"evaluation.deployment_candidate.{name} must be an integer >= {minimum}"
            )


def validate_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    storage = str(dataset.get("storage", "csv"))
    if storage not in {"csv", "parquet"}:
        raise ValueError("dataset.storage must be csv or parquet")
    shard_manifest = dataset.get("shard_manifest")
    if storage == "parquet":
        if not isinstance(shard_manifest, str) or not shard_manifest.strip():
            raise ValueError(
                "dataset.shard_manifest must be a non-empty path for parquet storage"
            )
        if Path(shard_manifest).suffix.casefold() != ".json":
            raise ValueError("dataset.shard_manifest must name a JSON manifest")
    elif shard_manifest is not None:
        raise ValueError("dataset.shard_manifest requires dataset.storage=parquet")
    for name in (
        "chunk_size",
        "record_batch_rows",
        "shard_target_rows",
        "quantile_sketch_capacity",
    ):
        value = dataset.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"dataset.{name} must be a positive integer")
    for name in ("max_rows_per_file", "max_rows_per_class", "max_loaded_rows"):
        value = dataset.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"dataset.{name} must be a positive integer or null")
    fractions = [
        float(config["split"]["train_fraction"]),
        float(config["split"]["validation_fraction"]),
        float(config["split"]["test_fraction"]),
    ]
    if any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must be positive and sum to 1.0")
    split_strategy = str(config["split"]["strategy"])
    if split_strategy not in {
        "random",
        "device",
        "attack",
        "time",
        "sequence",
        "block",
        "cross",
    }:
        raise ValueError("unsupported split.strategy")
    if config["split"].get("held_out_attacks") and split_strategy != "attack":
        raise ValueError("split.held_out_attacks requires split.strategy=attack")
    if config["split"].get("held_out_devices") and split_strategy != "device":
        raise ValueError("split.held_out_devices requires split.strategy=device")
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
    if config["model"]["type"] in {"fp32_mlp", "vanilla_bnn", "cost_aware_bnn"}:
        hidden_dims = list(config["model"].get("hidden_dims", []))
        if not hidden_dims or any(int(dimension) <= 0 for dimension in hidden_dims):
            raise ValueError("model.hidden_dims must contain at least one positive dimension")
    _finite_unit_interval(
        config["model"].get("minimum_active_fraction", 0.0),
        "model.minimum_active_fraction",
    )
    warmup_fraction = _finite_unit_interval(
        config["loss"].get("feature_penalty_warmup_fraction", 0.0),
        "loss.feature_penalty_warmup_fraction",
    )
    ramp_fraction = _finite_unit_interval(
        config["loss"].get("feature_penalty_ramp_fraction", 0.0),
        "loss.feature_penalty_ramp_fraction",
    )
    if warmup_fraction + ramp_fraction > 1.0:
        raise ValueError(
            "loss feature penalty warmup and ramp fractions must sum to at most 1"
        )
    thresholds = config["temporal"]["action_thresholds"]
    if len(thresholds) != 5 or list(thresholds) != sorted(thresholds):
        raise ValueError("temporal.action_thresholds must contain five increasing values")
    _validated_selection_weights(
        config["training"].get("selection_weights", {})
    )
    for device, risk in config["cascade"].get("device_criticality", {}).items():
        if not 0.0 <= float(risk) <= 1.0:
            raise ValueError(f"device criticality for {device!r} must be in [0, 1]")
    if int(config["temporal"].get("max_devices", 4096)) <= 0:
        raise ValueError("temporal.max_devices must be positive")
    if int(config["training"].get("checkpoint_every_epochs", 1)) <= 0:
        raise ValueError("training.checkpoint_every_epochs must be positive")
    for name, default in (
        ("checkpoint_every_steps", 1000),
        ("shuffle_buffer_rows", 262144),
    ):
        value = config["training"].get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{name} must be a positive integer")
    for target in config["evaluation"].get("fixed_fpr_targets", []):
        if not 0.0 < float(target) < 1.0:
            raise ValueError("evaluation.fixed_fpr_targets must be between 0 and 1")
    deployment_policy = config["evaluation"].get("deployment_candidate")
    _validate_deployment_candidate_policy(deployment_policy)
    if bool(deployment_policy["enabled"]):
        required_classes = set(deployment_policy["required_classes"])
        high_risk_classes = set(deployment_policy["high_risk_classes"])
        if not required_classes:
            raise ValueError(
                "evaluation.deployment_candidate.required_classes must not be empty"
            )
        if (
            not high_risk_classes
            or "benign" in high_risk_classes
            or not high_risk_classes.issubset(required_classes)
        ):
            raise ValueError(
                "evaluation.deployment_candidate.high_risk_classes must be a "
                "non-empty subset of required_classes without benign"
            )
        target = float(deployment_policy["fixed_fpr_target"])
        configured_targets = {
            float(value) for value in config["evaluation"].get("fixed_fpr_targets", [])
        }
        if target not in configured_targets:
            raise ValueError(
                "deployment fixed_fpr_target must appear in evaluation.fixed_fpr_targets"
            )
        if storage != "parquet":
            raise ValueError(
                "evaluation.deployment_candidate.enabled requires "
                "dataset.storage=parquet"
            )


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
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
        "packages": {},
        "determinism": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    distributions = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "joblib": "joblib",
        "PyYAML": "PyYAML",
        "torch": "torch",
    }
    for key, distribution in distributions.items():
        try:
            manifest["packages"][key] = version(distribution)
        except PackageNotFoundError:
            manifest["packages"][key] = None
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
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": (
                    torch.backends.cudnn.benchmark if hasattr(torch.backends, "cudnn") else None
                ),
                "cudnn_deterministic": (
                    torch.backends.cudnn.deterministic
                    if hasattr(torch.backends, "cudnn")
                    else None
                ),
            }
        )
    except ImportError:
        manifest["torch"] = None
    return manifest
