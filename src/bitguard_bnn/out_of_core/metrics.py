from __future__ import annotations

import heapq
import json
import math
import shutil
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray

from ..constants import CANONICAL_LABELS

_MAX_MERGE_FAN_IN = 32


class _CompensatedSum:
    """A mergeable Neumaier sum used for row-level floating totals."""

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated

    def add_array(self, values: np.ndarray) -> None:
        # fsum supplies a stable batch subtotal; Neumaier keeps separately
        # produced batches mergeable without retaining rows.
        self.add(math.fsum(np.asarray(values, dtype=np.float64).reshape(-1)))

    def merge(self, other: _CompensatedSum) -> None:
        self.add(other.total)
        self.add(other.correction)

    @property
    def value(self) -> float:
        return float(self.total + self.correction)


class _ClassScoreRuns:
    """Bounded exact `(score, target, row_uid)` sorted disk runs."""

    def __init__(self, root: Path, label_index: int, run_rows: int) -> None:
        self.root = root
        self.label_index = int(label_index)
        self.run_rows = int(run_rows)
        self.buffer: list[tuple[float, int, str]] = []
        self.paths: list[Path] = []
        self.merge_generation = 0
        self.positives = 0
        self.rows = 0

    def update(
        self, scores: np.ndarray, targets: np.ndarray, row_uids: np.ndarray
    ) -> None:
        for score, target, row_uid in zip(scores, targets, row_uids, strict=True):
            self.buffer.append((float(score), int(target), str(row_uid)))
            self.positives += int(target)
            self.rows += 1
            if len(self.buffer) >= self.run_rows:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.buffer.sort(key=lambda row: (-row[0], row[1], row[2]))
        path = self.root / f"class-{self.label_index:03d}-{len(self.paths):08d}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for score, target, row_uid in self.buffer:
                handle.write(
                    json.dumps(
                        [score, target, row_uid],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
        self.paths.append(path)
        self.buffer.clear()

    @staticmethod
    def _next_row(handle: TextIO) -> tuple[float, int, str] | None:
        line = handle.readline()
        if not line:
            return None
        value = json.loads(line)
        return float(value[0]), int(value[1]), str(value[2])

    @classmethod
    def _iter_paths(
        cls, paths: Sequence[Path]
    ) -> Iterator[tuple[float, int, str]]:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(path.open("r", encoding="utf-8"))
                for path in paths
            ]
            heap: list[tuple[float, int, str, int]] = []
            for index, handle in enumerate(handles):
                row = cls._next_row(handle)
                if row is not None:
                    score, target, row_uid = row
                    heapq.heappush(heap, (-score, target, row_uid, index))
            while heap:
                negative_score, target, row_uid, index = heapq.heappop(heap)
                yield -negative_score, target, row_uid
                row = cls._next_row(handles[index])
                if row is not None:
                    score, next_target, next_uid = row
                    heapq.heappush(
                        heap, (-score, next_target, next_uid, index)
                    )

    def _compact(self) -> None:
        while len(self.paths) > _MAX_MERGE_FAN_IN:
            following: list[Path] = []
            for group_index, start in enumerate(
                range(0, len(self.paths), _MAX_MERGE_FAN_IN)
            ):
                group = self.paths[start : start + _MAX_MERGE_FAN_IN]
                if len(group) == 1:
                    following.append(group[0])
                    continue
                output = self.root / (
                    f"class-{self.label_index:03d}-merge-{self.merge_generation:04d}-"
                    f"{group_index:08d}.jsonl"
                )
                with output.open("x", encoding="utf-8", newline="\n") as handle:
                    for score, target, row_uid in self._iter_paths(group):
                        handle.write(
                            json.dumps(
                                [score, target, row_uid],
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                        )
                        handle.write("\n")
                for path in group:
                    path.unlink()
                following.append(output)
            self.paths = following
            self.merge_generation += 1

    def iter_descending(self) -> Iterator[tuple[float, int, str]]:
        self.flush()
        self._compact()
        yield from self._iter_paths(self.paths)

    def copy_runs_from(self, other: _ClassScoreRuns) -> None:
        other.flush()
        for source in other.paths:
            destination = self.root / (
                f"class-{self.label_index:03d}-{len(self.paths):08d}-"
                f"{uuid.uuid4().hex}.jsonl"
            )
            shutil.copyfile(source, destination)
            self.paths.append(destination)
        self.positives += other.positives
        self.rows += other.rows


def _ordered_binary_metrics(runs: _ClassScoreRuns) -> tuple[float, float] | None:
    positives = int(runs.positives)
    negatives = int(runs.rows - positives)
    if positives == 0 or negatives == 0:
        return None

    true_positives = 0
    false_positives = 0
    old_recall = 0.0
    old_fpr = 0.0
    roc_area = 0.0
    average_precision = 0.0
    active_score: float | None = None
    group_positive = 0
    group_negative = 0

    def finish_group() -> None:
        nonlocal true_positives, false_positives, old_recall, old_fpr
        nonlocal roc_area, average_precision, group_positive, group_negative
        if group_positive == 0 and group_negative == 0:
            return
        true_positives += group_positive
        false_positives += group_negative
        recall = true_positives / positives
        fpr = false_positives / negatives
        roc_area += (recall + old_recall) * 0.5 * (fpr - old_fpr)
        average_precision += (recall - old_recall) * (
            true_positives / (true_positives + false_positives)
        )
        old_recall = recall
        old_fpr = fpr
        group_positive = 0
        group_negative = 0

    for score, target, _row_uid in runs.iter_descending():
        if active_score is None:
            active_score = score
        elif score != active_score:
            finish_group()
            active_score = score
        if target:
            group_positive += 1
        else:
            group_negative += 1
    finish_group()
    return float(roc_area), float(average_precision)


class StreamingClassificationMetrics:
    """Exact full-population metrics with bounded memory and disk score runs."""

    def __init__(
        self,
        *,
        probability_labels: Sequence[str],
        high_risk_labels: Sequence[str],
        temporary_directory: str | Path,
        score_run_rows: int = 131_072,
    ) -> None:
        labels = [str(label) for label in probability_labels]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("probability_labels must be non-empty and unique")
        if "benign" not in labels:
            raise ValueError("probability_labels must contain benign")
        if score_run_rows <= 0:
            raise ValueError("score_run_rows must be positive")
        self.probability_labels = labels
        self.high_risk_labels = {str(label) for label in high_risk_labels}
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.root = Path(temporary_directory) / f"metric-runs-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.score_run_rows = int(score_run_rows)
        class_count = len(labels)
        self.confusion: NDArray[np.int64] = np.zeros(
            (class_count, class_count), dtype=np.int64
        )
        self.rows = 0
        self.true_counts: NDArray[np.int64] = np.zeros(
            class_count, dtype=np.int64
        )
        self.pred_counts: NDArray[np.int64] = np.zeros(
            class_count, dtype=np.int64
        )
        self.high_risk_rows = 0
        self.high_risk_false_negatives = 0
        self.unknown_rows = 0
        self.unknown_correct = 0
        self.brier = _CompensatedSum()
        self.ece_counts: NDArray[np.int64] = np.zeros(10, dtype=np.int64)
        self.ece_confidence = [_CompensatedSum() for _ in range(10)]
        self.ece_correct: NDArray[np.int64] = np.zeros(10, dtype=np.int64)
        self.score_runs = [
            _ClassScoreRuns(self.root, index, self.score_run_rows)
            for index in range(class_count)
        ]
        self._closed = False

    def _encode(self, values: np.ndarray, name: str) -> np.ndarray:
        encoded: NDArray[np.int64] = np.empty(len(values), dtype=np.int64)
        for index, value in enumerate(np.asarray(values, dtype=str)):
            try:
                encoded[index] = self.label_to_index[str(value)]
            except KeyError as error:
                raise ValueError(f"{name} contains unknown label {value!r}") from error
        return encoded

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: np.ndarray,
        row_uids: np.ndarray,
    ) -> None:
        if self._closed:
            raise RuntimeError("metric accumulator is closed")
        true_values = np.asarray(y_true, dtype=str)
        pred_values = np.asarray(y_pred, dtype=str)
        scores = np.asarray(probabilities, dtype=np.float64)
        uids = np.asarray(row_uids, dtype=str)
        if true_values.ndim != 1 or pred_values.shape != true_values.shape:
            raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
        if scores.shape != (len(true_values), len(self.probability_labels)):
            raise ValueError("probabilities must align with rows and probability_labels")
        if uids.shape != true_values.shape:
            raise ValueError("row_uids must align with y_true")
        if not np.isfinite(scores).all():
            raise ValueError("probabilities must be finite")
        if len(set(uids.tolist())) != len(uids):
            raise ValueError("row_uids must be unique within each update")

        true_index = self._encode(true_values, "y_true")
        pred_index = self._encode(pred_values, "y_pred")
        np.add.at(self.confusion, (true_index, pred_index), 1)
        self.true_counts += np.bincount(
            true_index, minlength=len(self.probability_labels)
        )
        self.pred_counts += np.bincount(
            pred_index, minlength=len(self.probability_labels)
        )
        self.rows += len(true_values)

        high_risk = np.isin(true_values, tuple(self.high_risk_labels))
        self.high_risk_rows += int(np.count_nonzero(high_risk))
        self.high_risk_false_negatives += int(
            np.count_nonzero(high_risk & (pred_values == "benign"))
        )
        unknown = true_values == "unknown_like"
        self.unknown_rows += int(np.count_nonzero(unknown))
        self.unknown_correct += int(
            np.count_nonzero(unknown & (pred_values == "unknown_like"))
        )

        row_brier = np.square(scores).sum(axis=1)
        row_brier += 1.0 - 2.0 * scores[np.arange(len(scores)), true_index]
        self.brier.add_array(row_brier)

        confidence = scores.max(axis=1)
        probability_prediction = scores.argmax(axis=1)
        correct = probability_prediction == true_index
        # Preserve the established implementation's exact floating boundary
        # construction, including np.linspace representation at 0.6/0.7.
        for bin_index, lower in enumerate(np.linspace(0.0, 0.9, 10)):
            upper = lower + 0.1
            mask = (confidence >= lower) & (
                confidence < upper if upper < 1.0 else confidence <= upper
            )
            if not mask.any():
                continue
            self.ece_counts[bin_index] += int(np.count_nonzero(mask))
            self.ece_correct[bin_index] += int(np.count_nonzero(correct[mask]))
            self.ece_confidence[bin_index].add_array(confidence[mask])

        for class_index, run in enumerate(self.score_runs):
            run.update(scores[:, class_index], true_index == class_index, uids)

    def merge(self, other: StreamingClassificationMetrics) -> None:
        if self._closed or other._closed:
            raise RuntimeError("cannot merge a closed metric accumulator")
        if (
            self.probability_labels != other.probability_labels
            or self.high_risk_labels != other.high_risk_labels
        ):
            raise ValueError("metric accumulator contracts do not match")
        self.confusion += other.confusion
        self.true_counts += other.true_counts
        self.pred_counts += other.pred_counts
        self.rows += other.rows
        self.high_risk_rows += other.high_risk_rows
        self.high_risk_false_negatives += other.high_risk_false_negatives
        self.unknown_rows += other.unknown_rows
        self.unknown_correct += other.unknown_correct
        self.brier.merge(other.brier)
        self.ece_counts += other.ece_counts
        self.ece_correct += other.ece_correct
        for target_sum, source_sum in zip(
            self.ece_confidence, other.ece_confidence, strict=True
        ):
            target_sum.merge(source_sum)
        for target_runs, source_runs in zip(
            self.score_runs, other.score_runs, strict=True
        ):
            target_runs.copy_runs_from(source_runs)

    def _confusion_metrics(self) -> dict[str, Any]:
        observed = [
            label
            for label in CANONICAL_LABELS
            if label in self.label_to_index
            and (
                self.true_counts[self.label_to_index[label]]
                + self.pred_counts[self.label_to_index[label]]
                > 0
            )
        ]
        indexes = np.asarray([self.label_to_index[label] for label in observed])
        matrix = self.confusion[np.ix_(indexes, indexes)]
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        diagonal = np.diag(matrix)
        precision = np.divide(
            diagonal,
            predicted,
            out=np.zeros(len(observed), dtype=np.float64),
            where=predicted != 0,
        )
        recall = np.divide(
            diagonal,
            support,
            out=np.zeros(len(observed), dtype=np.float64),
            where=support != 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros(len(observed), dtype=np.float64),
            where=(precision + recall) != 0,
        )
        total = int(matrix.sum())
        correct = int(diagonal.sum())
        true_only_recall = recall[support != 0]
        numerator = float(correct * total - np.dot(support, predicted))
        denominator = math.sqrt(
            float(total * total - np.dot(predicted, predicted))
            * float(total * total - np.dot(support, support))
        )
        return {
            "accuracy": float(correct / total) if total else 0.0,
            "balanced_accuracy": (
                float(np.mean(true_only_recall)) if len(true_only_recall) else 0.0
            ),
            "macro_precision": float(np.mean(precision)) if len(precision) else 0.0,
            "macro_recall": float(np.mean(recall)) if len(recall) else 0.0,
            "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
            "mcc": float(numerator / denominator) if denominator else 0.0,
            "per_class": {
                label: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(observed)
            },
        }

    def finalize(
        self, operating_thresholds: Mapping[float, float] | None = None
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("metric accumulator is closed")
        metrics = self._confusion_metrics()
        metrics["high_risk_false_negative_rate"] = (
            self.high_risk_false_negatives / self.high_risk_rows
            if self.high_risk_rows
            else None
        )
        metrics["unknown_like_recall"] = (
            self.unknown_correct / self.unknown_rows if self.unknown_rows else None
        )
        auroc: dict[str, float] = {}
        auprc: dict[str, float] = {}
        for label, runs in zip(
            self.probability_labels, self.score_runs, strict=True
        ):
            ordered = _ordered_binary_metrics(runs)
            if ordered is not None:
                auroc[label], auprc[label] = ordered
        metrics["auroc_per_class"] = auroc
        metrics["auprc_per_class"] = auprc
        metrics["macro_auroc"] = float(np.mean(list(auroc.values()))) if auroc else None
        metrics["macro_auprc"] = float(np.mean(list(auprc.values()))) if auprc else None
        metrics["multiclass_brier_score"] = (
            self.brier.value / self.rows if self.rows else 0.0
        )
        ece = 0.0
        if self.rows:
            for index, count in enumerate(self.ece_counts):
                if count:
                    accuracy = self.ece_correct[index] / int(count)
                    confidence = self.ece_confidence[index].value / int(count)
                    ece += int(count) / self.rows * abs(accuracy - confidence)
        metrics["expected_calibration_error_10_bin"] = float(ece)

        fixed_fpr: dict[str, Any] = {}
        if operating_thresholds:
            normalized: list[tuple[float, float]] = []
            for raw_target, raw_threshold in sorted(operating_thresholds.items()):
                target = float(raw_target)
                threshold = float(raw_threshold)
                if not 0.0 < target < 1.0 or not np.isfinite(threshold):
                    raise ValueError(
                        "fixed-FPR operating points must contain valid targets and thresholds"
                    )
                normalized.append((target, threshold))
            benign_runs = self.score_runs[self.label_to_index["benign"]]
            benign_hits = [0] * len(normalized)
            attack_hits = [0] * len(normalized)
            benign_rows = int(benign_runs.positives)
            attack_rows = int(benign_runs.rows - benign_runs.positives)
            for benign_probability, is_benign, _uid in benign_runs.iter_descending():
                attack_score = 1.0 - benign_probability
                for index, (_target, threshold) in enumerate(normalized):
                    if attack_score >= threshold:
                        if is_benign:
                            benign_hits[index] += 1
                        else:
                            attack_hits[index] += 1
            fixed_fpr["threshold_source"] = "validation"
            for index, (target, threshold) in enumerate(normalized):
                suffix = f"{target:g}"
                fixed_fpr[f"attack_recall_at_benign_fpr_{suffix}"] = (
                    attack_hits[index] / attack_rows if attack_rows else None
                )
                fixed_fpr[f"observed_benign_fpr_at_target_{suffix}"] = (
                    benign_hits[index] / benign_rows if benign_rows else None
                )
                fixed_fpr[f"threshold_at_benign_fpr_{suffix}"] = threshold
        metrics["fixed_fpr"] = fixed_fpr
        return metrics

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        for runs in self.score_runs:
            runs.buffer.clear()
        shutil.rmtree(self.root, ignore_errors=False)

    def __enter__(self) -> StreamingClassificationMetrics:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()


__all__ = ["StreamingClassificationMetrics"]
