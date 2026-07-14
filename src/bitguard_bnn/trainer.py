from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from .cascade import (
    apply_boolean_fast_path,
    cascade_operation_summary,
    route_with_temporal_state,
    tune_boolean_fast_path,
    tune_exit_threshold,
)
from .config import (
    create_run_dir,
    environment_manifest,
    load_config,
    save_json,
    save_yaml,
    seed_everything,
)
from .constants import CANONICAL_LABELS
from .data import (
    DataSplit,
    LoadedDataset,
    load_dataset,
    make_cross_split,
    make_split,
    validate_labels,
)
from .metrics import (
    benchmark_torch_model,
    classification_metrics,
    confusion_frame,
    estimate_dense_operations,
    make_plots,
)
from .preprocess import FeaturePreprocessor, class_weights
from .state import replay_predictions


@dataclass
class NeuralFitResult:
    model: Any
    history: pd.DataFrame
    best_validation_score: float
    best_epoch: int


def _process_resource_summary(artifact: Path) -> dict[str, Any]:
    try:
        import resource

        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss_bytes: int | None = maximum_rss if sys.platform == "darwin" else maximum_rss * 1024
    except ImportError:
        rss_bytes = None
    return {
        "artifact_file_bytes": artifact.stat().st_size,
        "peak_process_rss_bytes_including_data_and_runtime": rss_bytes,
        "energy_per_decision_joules": None,
        "energy_note": "Measure on the target edge device with an external power monitor.",
    }


def _select_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _fit_neural(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
    teacher_model: Any | None = None,
) -> NeuralFitResult:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from .losses import BitGuardObjective
    from .models import clamp_binary_master_weights

    training_cfg = config["training"]
    device = _select_device(str(training_cfg.get("device", "auto")))
    model = model.to(device)
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
    weight_tensor = torch.from_numpy(weights).to(device) if config["loss"].get("class_weighted", True) else None
    objective = BitGuardObjective(config, weight_tensor, benign_index=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    batch_size = min(int(training_cfg["batch_size"]), len(x_train))
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    drop_last = len(dataset) > batch_size and len(dataset) % batch_size == 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    amp_enabled = bool(training_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    patience = int(training_cfg["patience"])
    best_metric = -math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    records: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("loss", "detection", "feature_cost", "fn", "fp")}
        seen = 0
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(features)
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_logits = teacher_model(features)
                else:
                    teacher_logits = None
                loss_output = objective(model, logits, target, teacher_logits)
            scaler.scale(loss_output.total).backward()
            if float(training_cfg.get("gradient_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            clamp_binary_master_weights(model)
            batch_rows = len(features)
            seen += batch_rows
            totals["loss"] += float(loss_output.total.detach()) * batch_rows
            totals["detection"] += float(loss_output.detection.detach()) * batch_rows
            totals["feature_cost"] += float(loss_output.feature_cost.detach()) * batch_rows
            totals["fn"] += float(loss_output.false_negative.detach()) * batch_rows
            totals["fp"] += float(loss_output.false_positive.detach()) * batch_rows
        scheduler.step()
        validation_probability = _predict_neural_probabilities(
            model, x_validation, int(training_cfg["batch_size"]), device
        )
        validation_prediction = validation_probability.argmax(axis=1)
        class_indices = list(range(validation_probability.shape[1]))
        validation_macro_f1 = float(
            f1_score(
                y_validation,
                validation_prediction,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        )
        per_class_auprc: list[float] = []
        for class_index in class_indices:
            binary_target = (y_validation == class_index).astype(np.int8)
            if binary_target.min() == binary_target.max():
                per_class_auprc.append(0.0)
            else:
                per_class_auprc.append(
                    float(
                        average_precision_score(
                            binary_target, validation_probability[:, class_index]
                        )
                    )
                )
        validation_macro_auprc = float(np.mean(per_class_auprc))
        attack_mask = y_validation != 0
        validation_attack_recall = (
            float(np.mean(validation_prediction[attack_mask] != 0)) if attack_mask.any() else 0.0
        )
        selection_weights = training_cfg["selection_weights"]
        validation_score = (
            float(selection_weights["macro_f1"]) * validation_macro_f1
            + float(selection_weights["macro_auprc"]) * validation_macro_auprc
            + float(selection_weights["attack_recall"]) * validation_attack_recall
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_macro_f1": validation_macro_f1,
            "validation_macro_auprc": validation_macro_auprc,
            "validation_attack_recall": validation_attack_recall,
            "validation_selection_score": validation_score,
        }
        record.update({f"train_{key}": value / max(seen, 1) for key, value in totals.items()})
        records.append(record)
        if validation_score > best_metric + 1e-6:
            best_metric = validation_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return NeuralFitResult(model, pd.DataFrame(records), best_metric, best_epoch)


def _predict_neural_probabilities(
    model: Any,
    values: np.ndarray,
    batch_size: int,
    device: Any | None = None,
) -> np.ndarray:
    import torch

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            outputs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _build_neural(
    config: dict[str, Any],
    preprocessor: FeaturePreprocessor,
    output_dim: int,
    *,
    input_indices: np.ndarray | None = None,
    hidden_dims: list[int] | None = None,
    force_bnn: bool = False,
) -> Any:
    from .models import build_model

    if input_indices is None:
        input_groups = preprocessor.input_groups
        input_dim = preprocessor.encoded_dimension
        costs = preprocessor.feature_costs
    else:
        all_groups = preprocessor.input_groups[input_indices]
        unique = sorted(set(int(item) for item in all_groups))
        remap = {old: new for new, old in enumerate(unique)}
        input_groups = np.asarray([remap[int(item)] for item in all_groups], dtype=np.int64)
        input_dim = len(input_indices)
        assert preprocessor.feature_costs is not None
        costs = preprocessor.feature_costs[unique]
    assert costs is not None
    return build_model(
        config,
        input_dim,
        output_dim,
        input_groups,
        costs,
        hidden_dims=hidden_dims,
        force_bnn=force_bnn,
    )


def _checkpoint(
    model: Any,
    config: dict[str, Any],
    preprocessor: FeaturePreprocessor,
    path: Path,
    output_labels: list[str],
    hidden_dims: list[int],
    input_indices: np.ndarray | None = None,
) -> None:
    import torch

    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "model_type": config["model"]["type"],
            "input_dim": len(input_indices) if input_indices is not None else preprocessor.encoded_dimension,
            "output_dim": len(output_labels),
            "output_labels": output_labels,
            "hidden_dims": list(hidden_dims),
            "dropout": float(config["model"].get("dropout", 0.0)),
            "binary_first_layer": bool(config["model"].get("binary_first_layer", True)),
            "input_indices": input_indices,
            "input_groups": preprocessor.input_groups if input_indices is None else preprocessor.input_groups[input_indices],
            "feature_costs": preprocessor.feature_costs,
        },
        path,
    )


def _fit_classical(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model
    if model_type == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed)
        weights = class_weights(y_train, int(y_train.max()) + 1)[y_train]
        model.fit(x_train, y_train, sample_weight=weights)
        return model
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "install the optional dependency with: pip install -e '.[xgboost]'"
            ) from error
        model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=seed,
        )
        weights = class_weights(y_train, int(y_train.max()) + 1)[y_train]
        model.fit(x_train, y_train, sample_weight=weights)
        return model
    raise ValueError(f"unsupported classical model: {model_type}")


def _to_full_probabilities(
    preprocessor: FeaturePreprocessor,
    known_probabilities: np.ndarray,
    unencoded_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels, _, active_plus_unknown = preprocessor.apply_open_set(
        known_probabilities, unencoded_values
    )
    full = np.zeros((len(known_probabilities), len(CANONICAL_LABELS)), dtype=np.float32)
    for index, label in enumerate([*preprocessor.active_labels, "unknown_like"]):
        full[:, CANONICAL_LABELS.index(label)] = active_plus_unknown[:, index]
    full /= np.maximum(full.sum(axis=1, keepdims=True), 1e-12)
    return labels, full


def _load_and_split(config: dict[str, Any]) -> tuple[LoadedDataset, DataSplit, list[str]]:
    source = load_dataset(config)
    validate_labels(source.frame)
    if config["split"]["strategy"] != "cross":
        return source, make_split(source, config), source.feature_columns
    cross_path = config["dataset"].get("cross_path")
    cross_type = config["dataset"].get("cross_type")
    if not cross_path or not cross_type:
        raise ValueError("cross split requires dataset.cross_path and dataset.cross_type")
    target_config = copy.deepcopy(config)
    target_config["dataset"]["type"] = cross_type
    target_config["dataset"]["path"] = cross_path
    target = load_dataset(target_config)
    validate_labels(target.frame)
    split, shared = make_cross_split(source, target, config)
    return source, split, shared


def run_training(config_path: str | Path) -> Path:
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    run_dir = create_run_dir(config)
    save_yaml(config, run_dir / "resolved_config.yaml")
    save_json(environment_manifest(), run_dir / "environment.json")
    _source, split, candidate_features = _load_and_split(config)
    save_json(split.manifest, run_dir / "split_manifest.json")
    preprocessor = FeaturePreprocessor(config).fit(split.train, candidate_features)
    preprocessor.calibrate_open_set(split.validation)
    preprocessor.save(run_dir / "preprocessor.joblib")
    save_json(preprocessor.feature_manifest(), run_dir / "feature_manifest.json")

    x_train = preprocessor.transform(split.train)
    x_validation = preprocessor.transform(split.validation)
    x_test = preprocessor.transform(split.test)
    x_validation_raw = preprocessor.transform_unencoded(split.validation)
    x_test_raw = preprocessor.transform_unencoded(split.test)
    y_train = preprocessor.encode_labels(split.train)
    y_validation = preprocessor.encode_labels(split.validation)
    if np.any(y_validation < 0):
        raise ValueError("validation contains a class absent from training; move it to held-out test")

    model_type = str(config["model"]["type"])
    history = pd.DataFrame()
    model_summary: dict[str, Any]
    if model_type in {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
    }:
        model = _fit_classical(model_type, x_train, y_train, seed)
        validation_known = model.predict_proba(x_validation).astype(np.float32)
        test_known = model.predict_proba(x_test).astype(np.float32)
        joblib.dump(model, run_dir / "best_model.joblib")
        model_summary = {
            "model_type": model_type,
            "artifact": "best_model.joblib",
            "classes": preprocessor.active_labels,
            **_process_resource_summary(run_dir / "best_model.joblib"),
        }
    else:
        from .models import parameter_summary

        teacher_model = None
        if float(config["loss"].get("distillation_alpha", 0.0)) > 0:
            if model_type == "fp32_mlp":
                raise ValueError("distillation_alpha requires a BNN student, not fp32_mlp")
            teacher_config = copy.deepcopy(config)
            teacher_config["model"]["type"] = "fp32_mlp"
            teacher_config["loss"]["distillation_alpha"] = 0.0
            teacher_config["loss"]["lambda_feature"] = 0.0
            teacher_model = _build_neural(
                teacher_config, preprocessor, len(preprocessor.active_labels)
            )
            teacher_fit = _fit_neural(
                teacher_model,
                x_train,
                y_train,
                x_validation,
                y_validation,
                class_weights(y_train, len(preprocessor.active_labels)),
                teacher_config,
            )
            teacher_model = teacher_fit.model
            teacher_fit.history.to_csv(run_dir / "teacher_training_history.csv", index=False)
            _checkpoint(
                teacher_model,
                teacher_config,
                preprocessor,
                run_dir / "teacher_model.pt",
                preprocessor.active_labels,
                list(config["model"]["hidden_dims"]),
            )
        model = _build_neural(config, preprocessor, len(preprocessor.active_labels))
        fit = _fit_neural(
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
            class_weights(y_train, len(preprocessor.active_labels)),
            config,
            teacher_model,
        )
        model = fit.model
        history = fit.history
        history.to_csv(run_dir / "training_history.csv", index=False)
        validation_known = _predict_neural_probabilities(
            model, x_validation, int(config["training"]["batch_size"])
        )
        test_known = _predict_neural_probabilities(
            model, x_test, int(config["training"]["batch_size"])
        )
        _checkpoint(
            model,
            config,
            preprocessor,
            run_dir / "best_model.pt",
            preprocessor.active_labels,
            list(config["model"]["hidden_dims"]),
        )
        model_summary = {
            "model_type": model_type,
            "artifact": "best_model.pt",
            "classes": preprocessor.active_labels,
            "best_validation_selection_score": fit.best_validation_score,
            "best_epoch": fit.best_epoch,
            **parameter_summary(model),
            **_process_resource_summary(run_dir / "best_model.pt"),
        }
        import torch

        sample = torch.from_numpy(x_test[:1]).to(next(model.parameters()).device)
        model_summary["latency"] = benchmark_torch_model(
            model,
            sample,
            int(config["evaluation"]["benchmark_warmup"]),
            int(config["evaluation"]["benchmark_repeats"]),
        )

    preprocessor.calibrate_confidence_threshold(validation_known, x_validation_raw)
    preprocessor.save(run_dir / "preprocessor.joblib")
    save_json(preprocessor.feature_manifest(), run_dir / "feature_manifest.json")
    save_yaml(preprocessor.config, run_dir / "calibrated_config.yaml")
    main_labels, test_full = _to_full_probabilities(preprocessor, test_known, x_test_raw)
    exit_stage = np.full(len(split.test), 2, dtype=np.int8)
    cascade_results: dict[str, Any] | None = None

    if bool(config["cascade"].get("enabled", False)):
        if model_type in {
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "xgboost",
        }:
            raise ValueError("cascade currently requires a neural Main model")
        tiny_budget = min(
            int(config["cascade"]["tiny_feature_budget"]), len(preprocessor.selected_features)
        )
        tiny_indices = preprocessor.encoder.encoded_indices_for_first(
            tiny_budget, len(preprocessor.selected_features)
        )
        tiny_config = copy.deepcopy(config)
        tiny_config["model"]["type"] = "vanilla_bnn"
        tiny_config["loss"]["distillation_alpha"] = 0.0
        tiny_model = _build_neural(
            tiny_config,
            preprocessor,
            2,
            input_indices=tiny_indices,
            hidden_dims=list(config["cascade"]["hidden_dims"]),
            force_bnn=True,
        )
        y_train_tiny = (y_train != 0).astype(np.int64)
        y_validation_tiny = (y_validation != 0).astype(np.int64)
        tiny_fit = _fit_neural(
            tiny_model,
            x_train[:, tiny_indices],
            y_train_tiny,
            x_validation[:, tiny_indices],
            y_validation_tiny,
            class_weights(y_train_tiny, 2),
            tiny_config,
        )
        tiny_model = tiny_fit.model
        tiny_fit.history.to_csv(run_dir / "tiny_training_history.csv", index=False)
        tiny_validation = _predict_neural_probabilities(
            tiny_model,
            x_validation[:, tiny_indices],
            int(config["training"]["batch_size"]),
        )
        calibration = tune_exit_threshold(
            tiny_validation[:, 0],
            split.validation["behavior_label"].to_numpy(),
            float(config["cascade"]["min_attack_recall"]),
            int(config["cascade"]["threshold_grid_size"]),
            float(config["cascade"]["false_negative_cost"]),
        )
        save_json(calibration.to_dict(), run_dir / "cascade_calibration.json")
        if bool(config["cascade"].get("boolean_fast_path_enabled", True)):
            boolean_calibration = tune_boolean_fast_path(
                split.validation,
                list(config["cascade"].get("boolean_fast_path_features", [])),
                float(config["cascade"]["min_attack_recall"]),
            )
        else:
            from .cascade import BooleanFastPathCalibration

            boolean_calibration = BooleanFastPathCalibration(False, [], {}, 1.0, 0.0)
        save_json(boolean_calibration.to_dict(), run_dir / "boolean_fast_path.json")
        boolean_test = apply_boolean_fast_path(
            split.test, boolean_calibration.upper_thresholds
        )
        tiny_test = _predict_neural_probabilities(
            tiny_model,
            x_test[:, tiny_indices],
            int(config["training"]["batch_size"]),
        )
        attack_prior = np.zeros(len(CANONICAL_LABELS), dtype=np.float64)
        train_attack_counts = split.train.loc[
            split.train["behavior_label"] != "benign", "behavior_label"
        ].value_counts()
        for label, count in train_attack_counts.items():
            if label in CANONICAL_LABELS and label != "unknown_like":
                attack_prior[CANONICAL_LABELS.index(label)] = float(count)
        routed, exit_stage, routing_summary = route_with_temporal_state(
            split.test[
                [
                    column
                    for column in ("source_file", "device_id", "timestamp", "sequence_index")
                    if column in split.test
                ]
            ],
            tiny_test[:, 0],
            test_full,
            CANONICAL_LABELS,
            calibration,
            config,
            attack_prior,
            boolean_test,
        )
        test_full = routed
        main_labels = np.where(exit_stage < 2, "benign", main_labels).astype(str)
        _checkpoint(
            tiny_model,
            tiny_config,
            preprocessor,
            run_dir / "tiny_model.pt",
            ["benign", "attack"],
            list(config["cascade"]["hidden_dims"]),
            tiny_indices,
        )
        main_ops = estimate_dense_operations(
            preprocessor.encoded_dimension,
            list(config["model"]["hidden_dims"]),
            len(preprocessor.active_labels),
        )
        tiny_ops = estimate_dense_operations(
            len(tiny_indices), list(config["cascade"]["hidden_dims"]), 2
        )
        true_evaluation = np.where(
            split.test["behavior_label"].isin(preprocessor.active_labels),
            split.test["behavior_label"],
            "unknown_like",
        )
        attack_mask = true_evaluation != "benign"
        cascade_results = {
            "calibration": calibration.to_dict(),
            "boolean_fast_path": boolean_calibration.to_dict(),
            "test_routing": routing_summary,
            "test_attack_escalation_recall": (
                float(np.mean(exit_stage[attack_mask] == 2)) if attack_mask.any() else None
            ),
            **cascade_operation_summary(
                exit_stage,
                tiny_ops,
                main_ops,
                len(boolean_calibration.features) if boolean_calibration.enabled else 0,
            ),
        }

    original_true = split.test["behavior_label"].astype(str).to_numpy()
    evaluation_true = np.where(
        np.isin(original_true, preprocessor.active_labels), original_true, "unknown_like"
    ).astype(str)
    metadata_columns = [
        column
        for column in (
            "row_uid",
            "dataset",
            "source_file",
            "sequence_index",
            "device_id",
            "raw_attack",
            "timestamp",
        )
        if column in split.test
    ]
    predictions = split.test[metadata_columns].copy()
    predictions["original_true_label"] = original_true
    predictions["true_label"] = evaluation_true
    predictions["predicted_label"] = main_labels
    predictions["exit_stage"] = exit_stage
    predictions["has_wall_clock_time"] = bool(
        split.manifest.get("provenance", split.manifest.get("target_provenance", {})).get(
            "has_wall_clock_time", False
        )
    )
    predictions["temporal_continuity"] = bool(split.manifest.get("temporal_continuity", False))
    for index, label in enumerate(CANONICAL_LABELS):
        predictions[f"prob_{label}"] = test_full[:, index]
    metrics = classification_metrics(
        evaluation_true,
        main_labels,
        CANONICAL_LABELS,
        test_full,
        list(config["evaluation"]["high_risk_labels"]),
    )
    confusion_frame(evaluation_true, main_labels).to_csv(run_dir / "confusion_matrix.csv")
    if bool(config["evaluation"].get("save_predictions", True)):
        predictions.to_csv(run_dir / "predictions.csv", index=False)
    plot_files: list[str] = []
    if bool(config["evaluation"].get("make_plots", True)):
        plot_files = make_plots(predictions, CANONICAL_LABELS, run_dir)
    operational: dict[str, Any] | None = None
    if bool(config["temporal"].get("enabled", False)):
        temporal_predictions, operational = replay_predictions(predictions, config)
        temporal_predictions.to_csv(run_dir / "temporal_predictions.csv", index=False)
        save_json(operational, run_dir / "operational_metrics.json")
    result = {
        "classification": metrics,
        "model": model_summary,
        "cascade": cascade_results,
        "operational": operational,
        "plots": plot_files,
        "research_validity": {
            "unknown_test_labels": "Any behavior absent from train is evaluated as unknown_like.",
            "native_cross_dataset_padding": False,
            "automatic_network_action": False,
            "pytorch_bnn_speed_claim": False,
        },
    }
    save_json(model_summary, run_dir / "model_summary.json")
    save_json(result, run_dir / "metrics.json")
    return run_dir
