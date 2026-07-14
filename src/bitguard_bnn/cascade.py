from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .state import TemporalSecurityStateMachine


@dataclass
class CascadeCalibration:
    exit_threshold: float
    attack_escalation_recall: float
    benign_early_exit_ratio: float
    overall_early_exit_ratio: float
    validation_rows: int
    false_negative_cost: float
    formula: str = "p_benign - (1-p_benign) - device_risk - temporal_risk - C_FN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BooleanFastPathCalibration:
    enabled: bool
    features: list[str]
    upper_thresholds: dict[str, float]
    attack_escalation_recall: float
    benign_early_exit_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tune_boolean_fast_path(
    validation: pd.DataFrame,
    features: list[str],
    min_attack_recall: float,
) -> BooleanFastPathCalibration:
    available = [feature for feature in features if feature in validation]
    if not available:
        return BooleanFastPathCalibration(False, [], {}, 1.0, 0.0)
    benign = validation["behavior_label"].astype(str).to_numpy() == "benign"
    attack = ~benign
    if not benign.any() or not attack.any():
        return BooleanFastPathCalibration(False, available, {}, 1.0, 0.0)
    best: tuple[float, dict[str, float], float] | None = None
    for quantile in np.linspace(0.50, 0.99, 50):
        thresholds = {
            feature: float(validation.loc[benign, feature].quantile(quantile))
            for feature in available
        }
        mask = apply_boolean_fast_path(validation, thresholds)
        attack_recall = float(np.mean(~mask[attack]))
        if attack_recall + 1e-12 < min_attack_recall:
            continue
        benign_exit = float(np.mean(mask[benign]))
        if best is None or benign_exit > best[0]:
            best = (benign_exit, thresholds, attack_recall)
    if best is None:
        return BooleanFastPathCalibration(False, available, {}, 1.0, 0.0)
    return BooleanFastPathCalibration(True, available, best[1], best[2], best[0])


def apply_boolean_fast_path(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    if not thresholds:
        return np.zeros(len(frame), dtype=bool)
    mask = np.ones(len(frame), dtype=bool)
    for feature, threshold in thresholds.items():
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy()
        mask &= np.isfinite(values) & (values <= threshold)
    return mask


def exit_scores(
    benign_probability: np.ndarray,
    *,
    temporal_risk: np.ndarray | float = 0.0,
    device_risk: np.ndarray | float = 0.0,
    false_negative_cost: float = 0.0,
) -> np.ndarray:
    benign_probability = np.asarray(benign_probability, dtype=np.float64)
    return (
        benign_probability
        - (1.0 - benign_probability)
        - np.asarray(temporal_risk, dtype=np.float64)
        - np.asarray(device_risk, dtype=np.float64)
        - float(false_negative_cost)
    )


def tune_exit_threshold(
    benign_probability: np.ndarray,
    true_labels: np.ndarray,
    min_attack_recall: float,
    grid_size: int,
    false_negative_cost: float,
) -> CascadeCalibration:
    # Calibrate a base threshold first. The runtime FN cost is then subtracted
    # without re-tuning, so increasing the cost can only escalate more rows.
    scores = exit_scores(benign_probability, false_negative_cost=0.0)
    truth = np.asarray(true_labels, dtype=str)
    attack = truth != "benign"
    benign = ~attack
    if not attack.any() or not benign.any():
        raise ValueError("cascade calibration requires benign and attack validation rows")
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(float(scores.min()) - 1e-6, float(scores.max()) + 1e-6, grid_size),
                scores,
            ]
        )
    )
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        early_exit = scores >= threshold
        attack_recall = float(np.mean(~early_exit[attack]))
        if attack_recall + 1e-12 < min_attack_recall:
            continue
        benign_exit = float(np.mean(early_exit[benign]))
        overall_exit = float(np.mean(early_exit))
        candidate = (benign_exit, overall_exit, -float(threshold), attack_recall)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        threshold = float(scores.max()) + 1e-6
        return CascadeCalibration(
            threshold,
            1.0,
            0.0,
            0.0,
            len(scores),
            false_negative_cost,
        )
    benign_exit, overall_exit, negative_threshold, attack_recall = best
    return CascadeCalibration(
        -negative_threshold,
        attack_recall,
        benign_exit,
        overall_exit,
        len(scores),
        false_negative_cost,
    )


def route_with_temporal_state(
    metadata: pd.DataFrame,
    tiny_benign_probability: np.ndarray,
    main_probabilities: np.ndarray,
    probability_labels: list[str],
    calibration: CascadeCalibration,
    config: dict[str, Any],
    attack_prior: np.ndarray | None = None,
    boolean_fast_path: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Route in chronological order, reading previous state before classification."""

    if len(metadata) != len(tiny_benign_probability) or len(metadata) != len(main_probabilities):
        raise ValueError("cascade arrays have inconsistent lengths")
    machine = TemporalSecurityStateMachine(config)
    cfg = config["cascade"]
    use_temporal = bool(config["temporal"].get("enabled", False)) and bool(
        cfg.get("use_temporal_state", True)
    )
    order_frame = metadata.copy()
    order_frame["__position"] = np.arange(len(order_frame))
    if "timestamp" in order_frame and order_frame["timestamp"].notna().all():
        order = order_frame.sort_values(["timestamp", "device_id", "__position"], kind="stable")[
            "__position"
        ].to_numpy()
    elif "sequence_index" in order_frame:
        order = order_frame.sort_values(["device_id", "sequence_index", "__position"], kind="stable")[
            "__position"
        ].to_numpy()
    else:
        order = np.arange(len(order_frame))
    output = np.zeros_like(main_probabilities, dtype=np.float32)
    exit_stage = np.full(len(metadata), 2, dtype=np.int8)
    scores = np.empty(len(metadata), dtype=np.float32)
    boolean_fast_path = (
        np.zeros(len(metadata), dtype=bool)
        if boolean_fast_path is None
        else np.asarray(boolean_fast_path, dtype=bool)
    )
    benign_index = probability_labels.index("benign")
    unknown_index = probability_labels.index("unknown_like")
    if attack_prior is None:
        attack_prior = np.ones(len(probability_labels), dtype=np.float64)
        attack_prior[[benign_index, unknown_index]] = 0.0
    attack_prior = np.asarray(attack_prior, dtype=np.float64)
    attack_prior[[benign_index, unknown_index]] = 0.0
    if attack_prior.sum() <= 0:
        raise ValueError("cascade attack prior needs at least one known attack class")
    attack_prior /= attack_prior.sum()
    temporal_coefficient = float(cfg.get("temporal_penalty", 0.30))
    device_default = float(cfg.get("device_criticality_default", 0.0))
    criticality = {str(key): float(value) for key, value in cfg.get("device_criticality", {}).items()}
    for position in order:
        row = order_frame.iloc[int(position)]
        state_key = f"{row.get('source_file', 'default_episode')}::{row['device_id']}"
        suspicion = machine.temporal_suspicion(state_key) if use_temporal else 0.0
        timestamp = row.get("timestamp")
        timestamp_value = float(timestamp) if timestamp is not None and pd.notna(timestamp) else None
        if boolean_fast_path[position] and suspicion <= 0.0:
            exit_stage[position] = 0
            output[position, benign_index] = 1.0
            scores[position] = 1.0
            if use_temporal:
                machine.update(state_key, {"benign": 1.0}, timestamp_value)
            continue
        score = exit_scores(
            np.asarray([tiny_benign_probability[position]]),
            temporal_risk=temporal_coefficient * suspicion,
            device_risk=criticality.get(str(row["device_id"]), device_default),
            false_negative_cost=calibration.false_negative_cost,
        )[0]
        scores[position] = score
        if score >= calibration.exit_threshold:
            exit_stage[position] = 1
            output[position, benign_index] = float(tiny_benign_probability[position])
            output[position] += float(1.0 - tiny_benign_probability[position]) * attack_prior
        else:
            output[position] = main_probabilities[position]
        probabilities = {
            label: float(output[position, index]) for index, label in enumerate(probability_labels)
        }
        if use_temporal:
            machine.update(state_key, probabilities, timestamp_value)
    summary = {
        "early_exit_ratio": float(np.mean(exit_stage == 1)),
        "boolean_fast_path_ratio": float(np.mean(exit_stage == 0)),
        "total_early_exit_ratio": float(np.mean(exit_stage < 2)),
        "main_model_ratio": float(np.mean(exit_stage == 2)),
        "mean_exit_score": float(scores.mean()),
        "temporal_state_read_before_route": True,
        "temporal_state_enabled": use_temporal,
        "early_exit_attack_residual": "distributed over known attack classes using train-only priors",
        "false_negative_cost_note": (
            "Threshold is calibrated on the base score; runtime FN cost is applied afterward, "
            "so it conservatively increases escalation."
        ),
    }
    return output, exit_stage, summary


def cascade_operation_summary(
    exit_stage: np.ndarray,
    tiny_operations: int,
    main_operations: int,
    boolean_operations: int = 0,
) -> dict[str, Any]:
    stages = np.asarray(exit_stage)
    tiny_ratio = float(np.mean(stages >= 1))
    main_ratio = float(np.mean(stages == 2))
    average = float(
        boolean_operations + tiny_ratio * tiny_operations + main_ratio * main_operations
    )
    full = float(boolean_operations + tiny_operations + main_operations)
    return {
        "boolean_comparisons_per_row": int(boolean_operations),
        "tiny_operations_per_row": int(tiny_operations),
        "main_operations_per_escalated_row": int(main_operations),
        "average_estimated_dense_equivalent_operations_per_row": average,
        "always_run_both_operations_per_row": full,
        "estimated_operation_reduction_ratio": float(1.0 - average / max(full, 1.0)),
        "boolean_fast_path_ratio": float(np.mean(stages == 0)),
    }
