from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .constants import CANONICAL_LABELS


def calibrate_fixed_fpr_thresholds(
    y_true: np.ndarray,
    probability_labels: list[str],
    probabilities: np.ndarray,
    target_fprs: Sequence[float] = (1e-2, 1e-3),
) -> dict[float, float]:
    """Calibrate attack-score thresholds from benign validation examples only."""
    y_true = np.asarray(y_true, dtype=str)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities must be a two-dimensional array aligned with y_true")
    if probabilities.shape[1] != len(probability_labels):
        raise ValueError("probability_labels must match the probability columns")
    if "benign" not in probability_labels:
        raise ValueError("fixed-FPR calibration requires a benign probability column")
    benign_mask = y_true == "benign"
    if not benign_mask.any():
        raise ValueError("fixed-FPR calibration requires benign validation examples")

    benign_column = probability_labels.index("benign")
    benign_attack_scores = 1.0 - probabilities[benign_mask, benign_column]
    thresholds: dict[float, float] = {}
    for value in target_fprs:
        target_fpr = float(value)
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target FPR values must be between 0 and 1")
        allowed_false_positives = int(
            np.floor(target_fpr * len(benign_attack_scores) + 1e-12)
        )
        descending = np.sort(benign_attack_scores)[::-1]
        if allowed_false_positives == 0:
            threshold = np.nextafter(descending[0], np.inf)
        else:
            threshold = descending[allowed_false_positives - 1]
            if int(np.count_nonzero(benign_attack_scores >= threshold)) > allowed_false_positives:
                threshold = np.nextafter(threshold, np.inf)
        thresholds[target_fpr] = float(threshold)
    return thresholds


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_labels: list[str],
    probabilities: np.ndarray,
    high_risk_labels: list[str],
    operating_thresholds: Mapping[float, float] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != y_true.shape[0]:
        raise ValueError("probabilities must be a two-dimensional array aligned with y_true")
    if probabilities.shape[1] != len(probability_labels):
        raise ValueError("probability_labels must match the probability columns")
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred must be aligned with y_true")
    if not set(y_true).issubset(probability_labels):
        raise ValueError("every true label must have a probability column")
    labels = [label for label in CANONICAL_LABELS if label in set(y_true) | set(y_pred)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }
    risk_mask = np.isin(y_true, high_risk_labels)
    metrics["high_risk_false_negative_rate"] = (
        float(np.mean(y_pred[risk_mask] == "benign")) if risk_mask.any() else None
    )
    unknown_mask = y_true == "unknown_like"
    metrics["unknown_like_recall"] = (
        float(np.mean(y_pred[unknown_mask] == "unknown_like")) if unknown_mask.any() else None
    )
    auroc: dict[str, float] = {}
    auprc: dict[str, float] = {}
    for column_index, label in enumerate(probability_labels):
        target = (y_true == label).astype(np.int8)
        if target.min(initial=0) == target.max(initial=0):
            continue
        try:
            auroc[label] = float(roc_auc_score(target, probabilities[:, column_index]))
            auprc[label] = float(average_precision_score(target, probabilities[:, column_index]))
        except ValueError:
            continue
    metrics["auroc_per_class"] = auroc
    metrics["auprc_per_class"] = auprc
    metrics["macro_auroc"] = float(np.mean(list(auroc.values()))) if auroc else None
    metrics["macro_auprc"] = float(np.mean(list(auprc.values()))) if auprc else None
    label_to_column = {label: index for index, label in enumerate(probability_labels)}
    true_columns = np.asarray([label_to_column[label] for label in y_true], dtype=np.int64)
    one_hot = np.eye(len(probability_labels), dtype=np.float64)[true_columns]
    metrics["multiclass_brier_score"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    probability_prediction = probabilities.argmax(axis=1)
    correct = probability_prediction == true_columns
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    metrics["expected_calibration_error_10_bin"] = ece
    benign_probability = probabilities[:, label_to_column["benign"]]
    attack_score = 1.0 - benign_probability
    benign_mask = y_true == "benign"
    attack_mask = ~benign_mask
    fixed_fpr: dict[str, Any] = {}
    if operating_thresholds:
        fixed_fpr["threshold_source"] = "validation"
        for raw_target_fpr, raw_threshold in sorted(operating_thresholds.items()):
            target_fpr = float(raw_target_fpr)
            threshold = float(raw_threshold)
            if not 0.0 < target_fpr < 1.0 or not np.isfinite(threshold):
                raise ValueError("fixed-FPR operating points must contain valid targets and thresholds")
            suffix = f"{target_fpr:g}"
            fixed_fpr[f"attack_recall_at_benign_fpr_{suffix}"] = (
                float(np.mean(attack_score[attack_mask] >= threshold))
                if attack_mask.any()
                else None
            )
            fixed_fpr[f"observed_benign_fpr_at_target_{suffix}"] = (
                float(np.mean(attack_score[benign_mask] >= threshold))
                if benign_mask.any()
                else None
            )
            fixed_fpr[f"threshold_at_benign_fpr_{suffix}"] = threshold
    metrics["fixed_fpr"] = fixed_fpr
    return metrics


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = [label for label in CANONICAL_LABELS if label in set(y_true) | set(y_pred)]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=[f"true_{item}" for item in labels], columns=[f"pred_{item}" for item in labels])


def make_plots(
    predictions: pd.DataFrame,
    probability_labels: list[str],
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    except ImportError:
        return []
    written: list[str] = []
    matrix = confusion_frame(
        predictions["true_label"].to_numpy(), predictions["predicted_label"].to_numpy()
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greys", cbar=False, ax=axis)
    axis.set_title("Confusion matrix")
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.name))
    fig_pr, axis_pr = plt.subplots(figsize=(8, 6))
    fig_roc, axis_roc = plt.subplots(figsize=(8, 6))
    plotted_pr = False
    plotted_roc = False
    for label in probability_labels:
        target = (predictions["true_label"] == label).astype(int)
        if target.nunique() < 2:
            continue
        probability = predictions[f"prob_{label}"]
        PrecisionRecallDisplay.from_predictions(target, probability, name=label, ax=axis_pr)
        RocCurveDisplay.from_predictions(target, probability, name=label, ax=axis_roc)
        plotted_pr = plotted_roc = True
    for fig, axis, name, plotted in (
        (fig_pr, axis_pr, "precision_recall.png", plotted_pr),
        (fig_roc, axis_roc, "roc.png", plotted_roc),
    ):
        if plotted:
            axis.set_title(name.replace("_", " ").replace(".png", "").title())
            fig.tight_layout()
            fig.savefig(output_dir / name, dpi=180)
            written.append(name)
        plt.close(fig)
    return written


def estimate_dense_operations(input_dim: int, hidden_dims: list[int], output_dim: int) -> int:
    dimensions = [input_dim, *hidden_dims, output_dim]
    return int(sum(left * right for left, right in zip(dimensions[:-1], dimensions[1:])))


def benchmark_torch_model(
    model: Any,
    sample: Any,
    warmup: int,
    repeats: int,
    synchronize_cuda: bool = True,
) -> dict[str, Any]:
    import torch

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if synchronize_cuda and sample.is_cuda:
            torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(sample)
            if synchronize_cuda and sample.is_cuda:
                torch.cuda.synchronize()
            timings.append((time.perf_counter_ns() - start) / 1_000.0)
    values = np.asarray(timings, dtype=np.float64)
    return {
        "backend": "pytorch_float_tensor_execution",
        "batch_size": int(sample.shape[0]),
        "p50_microseconds": float(np.percentile(values, 50)),
        "p95_microseconds": float(np.percentile(values, 95)),
        "mean_microseconds": float(values.mean()),
        "repeats": repeats,
        "warning": "PyTorch sign layers do not prove XNOR/popcount edge latency.",
    }
