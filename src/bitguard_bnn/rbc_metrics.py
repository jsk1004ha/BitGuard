"""Evaluation helpers for the BitGuard-RBC detection-first protocol.

Calibration accepts benign scores from the independent calibration split.  Evaluation
accepts the separately held-out test split and never adjusts the supplied threshold.
All deployed decisions use the protocol's strict ``score > threshold`` rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta, binom


def _one_dimensional(values: Sequence[Any] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def calibrate_benign_threshold(
    benign_scores: Sequence[float] | np.ndarray,
    target_fpr: float,
    *,
    method: str = "empirical",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Choose a threshold using only independent benign calibration scores.

    ``empirical`` selects the lowest observed threshold whose strict empirical FPR is
    no greater than ``target_fpr``. ``np_order_statistic`` selects the Neyman-Pearson
    order statistic whose probability of exceeding the population FPR target is at
    most ``1 - confidence`` under the i.i.d. calibration assumption.
    """

    scores = _one_dimensional(benign_scores, "benign_scores").astype(np.float64)
    target_fpr = float(target_fpr)
    confidence = float(confidence)
    if scores.size == 0:
        raise ValueError("benign_scores must not be empty")
    if not np.isfinite(scores).all():
        raise ValueError("benign_scores must contain only finite values")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if method not in {"empirical", "np_order_statistic"}:
        raise ValueError("method must be 'empirical' or 'np_order_statistic'")

    n_benign = int(scores.size)
    result: dict[str, Any] = {
        "status": "ok",
        "method": method,
        "threshold_rule": "score > threshold",
        "target_fpr": target_fpr,
        "confidence": confidence,
        "n_benign": n_benign,
    }

    if method == "empirical":
        allowed = int(np.floor(target_fpr * n_benign))
        # With a strict decision, the (allowed + 1)-th score in descending order
        # leaves at most ``allowed`` observations strictly above the threshold.
        threshold = float(np.partition(scores, n_benign - allowed - 1)[n_benign - allowed - 1])
        order_index: int | None = n_benign - allowed
    else:
        violation_probability = 1.0 - confidence
        quantile_probability = 1.0 - target_fpr
        required_n = int(np.ceil(np.log(violation_probability) / np.log(quantile_probability)))
        result["minimum_n_benign"] = required_n
        if n_benign < required_n:
            result.update(
                {
                    "status": "insufficient_n",
                    "threshold": None,
                    "order_statistic_index": None,
                    "empirical_false_positives": None,
                    "empirical_fpr": None,
                }
            )
            return result

        # Find the smallest 1-based ascending order statistic k satisfying
        # P[Binomial(n, 1-alpha) >= k] <= 1-confidence.
        order_index = int(binom.ppf(confidence, n_benign, quantile_probability)) + 1
        order_index = min(max(order_index, 1), n_benign)
        while order_index > 1 and binom.sf(
            order_index - 2, n_benign, quantile_probability
        ) <= violation_probability:
            order_index -= 1
        while order_index <= n_benign and binom.sf(
            order_index - 1, n_benign, quantile_probability
        ) > violation_probability:
            order_index += 1
        if order_index > n_benign:
            # Protect the explicit contract from floating-point boundary behavior.
            result.update(
                {
                    "status": "insufficient_n",
                    "threshold": None,
                    "order_statistic_index": None,
                    "empirical_false_positives": None,
                    "empirical_fpr": None,
                }
            )
            return result
        threshold = float(np.partition(scores, order_index - 1)[order_index - 1])
        result["violation_probability"] = float(
            binom.sf(order_index - 1, n_benign, quantile_probability)
        )

    false_positives = int(np.count_nonzero(scores > threshold))
    result.update(
        {
            "threshold": threshold,
            "order_statistic_index": order_index,
            "empirical_false_positives": false_positives,
            "empirical_fpr": float(false_positives / n_benign),
        }
    )
    return result


def binomial_fpr_upper_bound(
    false_positives: int,
    n_benign: int,
    *,
    confidence: float = 0.95,
    iid: bool,
) -> dict[str, Any]:
    """Return a one-sided exact binomial FPR bound only for i.i.d. tests."""

    false_positives = int(false_positives)
    n_benign = int(n_benign)
    confidence = float(confidence)
    if n_benign < 0 or not 0 <= false_positives <= n_benign:
        raise ValueError("false_positives must be between 0 and n_benign")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not iid:
        return {
            "status": "not_applicable_non_iid",
            "confidence": confidence,
            "upper_bound": None,
        }
    if n_benign == 0:
        return {"status": "insufficient_n", "confidence": confidence, "upper_bound": None}
    upper = (
        1.0
        if false_positives == n_benign
        else float(beta.ppf(confidence, false_positives + 1, n_benign - false_positives))
    )
    return {"status": "ok", "confidence": confidence, "upper_bound": upper}


def _fixed_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        true = y_true == label
        predicted = y_pred == label
        tp = int(np.count_nonzero(true & predicted))
        fp = int(np.count_nonzero(~true & predicted))
        fn = int(np.count_nonzero(true & ~predicted))
        precision = _rate(tp, tp + fp) or 0.0
        recall = _rate(tp, tp + fn) or 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(np.count_nonzero(true)),
        }
    return {
        "labels": list(labels),
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "per_class": per_class,
    }


def evaluate_rbc(
    scores: Sequence[float] | np.ndarray,
    true_labels: Sequence[str] | np.ndarray,
    threshold: float,
    *,
    benign_label: str = "benign",
    known_attack_labels: Sequence[str],
    unknown_attack_labels: Sequence[str] = (),
    type_predictions: Sequence[str] | np.ndarray | None = None,
    groups: Mapping[str, Sequence[Any] | np.ndarray] | None = None,
    unknown_prediction_label: str = "unknown",
    test_iid: bool = False,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Evaluate a fixed RBC threshold on a held-out labeled test set."""

    score_array = _one_dimensional(scores, "scores").astype(np.float64)
    truth = _one_dimensional(true_labels, "true_labels").astype(str)
    if score_array.shape != truth.shape:
        raise ValueError("scores and true_labels must be aligned")
    if not np.isfinite(score_array).all() or not np.isfinite(float(threshold)):
        raise ValueError("scores and threshold must be finite")
    known = tuple(str(label) for label in known_attack_labels)
    unknown = tuple(str(label) for label in unknown_attack_labels)
    fixed_labels = (str(benign_label), *known, *unknown)
    if len(fixed_labels) != len(set(fixed_labels)):
        raise ValueError("benign, known attack, and unknown attack labels must be disjoint")
    unexpected = sorted(set(truth) - set(fixed_labels))
    if unexpected:
        raise ValueError(f"true_labels contain labels outside the fixed label set: {unexpected}")

    detected = score_array > float(threshold)
    benign_mask = truth == benign_label
    attack_mask = ~benign_mask
    tp = int(np.count_nonzero(detected & attack_mask))
    fn = int(np.count_nonzero(~detected & attack_mask))
    fp = int(np.count_nonzero(detected & benign_mask))
    tn = int(np.count_nonzero(~detected & benign_mask))
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "threshold_rule": "score > threshold",
        "detection": {
            "tpr": _rate(tp, tp + fn),
            "fnr": _rate(fn, tp + fn),
            "fpr": _rate(fp, fp + tn),
            "precision": _rate(tp, tp + fp),
            "support_attack": tp + fn,
            "support_benign": fp + tn,
        },
        "benign_only": {
            "false_positives": fp,
            "support": fp + tn,
            "fpr": _rate(fp, fp + tn),
        },
        "test_fpr_upper_bound": binomial_fpr_upper_bound(
            fp, fp + tn, confidence=confidence, iid=test_iid
        ),
    }

    operational_types: np.ndarray | None = None
    if type_predictions is not None:
        predicted_types = _one_dimensional(type_predictions, "type_predictions").astype(str)
        if predicted_types.shape != truth.shape:
            raise ValueError("type_predictions and true_labels must be aligned")
        operational_types = np.where(detected, predicted_types, benign_label)
        known_mask = np.isin(truth, known)
        detected_known_mask = known_mask & detected
        result["attack_type"] = {
            "conditional_on_detection": _fixed_class_metrics(
                truth[detected_known_mask], operational_types[detected_known_mask], known
            ),
            "end_to_end": _fixed_class_metrics(
                truth[known_mask], operational_types[known_mask], known
            ),
        }

    unknown_mask = np.isin(truth, unknown)
    known_mask = np.isin(truth, known)
    result["unknown_attack"] = {
        "support": int(np.count_nonzero(unknown_mask)),
        "detection_rate": (
            float(np.mean(detected[unknown_mask])) if unknown_mask.any() else None
        ),
        "rejection_rate": (
            float(np.mean(operational_types[unknown_mask] == unknown_prediction_label))
            if operational_types is not None and unknown_mask.any()
            else None
        ),
        "known_to_unknown_rate": (
            float(np.mean(operational_types[known_mask] == unknown_prediction_label))
            if operational_types is not None and known_mask.any()
            else None
        ),
        "benign_to_unknown_rate": (
            float(np.mean(operational_types[benign_mask] == unknown_prediction_label))
            if operational_types is not None and benign_mask.any()
            else None
        ),
    }

    grouped: dict[str, Any] = {}
    for dimension, raw_values in (groups or {}).items():
        values = _one_dimensional(raw_values, f"groups[{dimension!r}]")
        if values.shape != truth.shape:
            raise ValueError(f"groups[{dimension!r}] and true_labels must be aligned")
        rows: dict[str, dict[str, Any]] = {}
        for value in sorted({str(item) for item in values[attack_mask]}):
            mask = attack_mask & (values.astype(str) == value)
            support = int(np.count_nonzero(mask))
            tpr = float(np.mean(detected[mask]))
            rows[value] = {"tpr": tpr, "fnr": 1.0 - tpr, "support": support}
        grouped[str(dimension)] = {
            "groups": rows,
            "macro_tpr": (
                float(np.mean([row["tpr"] for row in rows.values()])) if rows else None
            ),
            "worst_tpr": min((row["tpr"] for row in rows.values()), default=None),
            "worst_fnr": max((row["fnr"] for row in rows.values()), default=None),
        }
    result["grouped_detection"] = grouped
    return result
