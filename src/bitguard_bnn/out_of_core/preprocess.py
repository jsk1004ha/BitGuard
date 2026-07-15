"""Train-only, bounded-memory preprocessing for prepared full datasets."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..constants import KNOWN_LABELS
from ..preprocess import FeaturePreprocessor, IdentityScaler
from .quantiles import PriorityRowSketch


STREAMING_PREPROCESS_VERSION = 1
_PHASE_INSPECT = 1
_PHASE_ANOVA = 2
_PHASE_CALIBRATE = 3


def _as_numeric_matrix(values: object, rows: int, width: int) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape != (rows, width):
        raise ValueError(
            f"values must have shape ({rows}, {width}); received {raw.shape}"
        )
    if raw.dtype.kind not in "iuf":
        raise TypeError("values must be numeric")
    return raw.astype(np.float64, copy=False)


def _row_value_digest(label: str, values: NDArray[np.float64]) -> str:
    tokens: list[str] = []
    for value in values:
        converted = float(value)
        if math.isnan(converted):
            tokens.append("nan")
        elif converted == math.inf:
            tokens.append("+inf")
        elif converted == -math.inf:
            tokens.append("-inf")
        else:
            tokens.append(converted.hex())
    encoded = json.dumps(
        [label, tokens], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ClassSufficientStatistics:
    """Mergeable per-class count/mean/M2 feature statistics."""

    def __init__(self, labels: Sequence[str], width: int) -> None:
        self.labels = tuple(str(label) for label in labels)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be non-empty and unique")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        self.width = width
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        self.counts: NDArray[np.int64] = np.zeros(len(self.labels), dtype=np.int64)
        self.means: NDArray[np.float64] = np.zeros(
            (len(self.labels), width), dtype=np.float64
        )
        self.m2: NDArray[np.float64] = np.zeros_like(self.means)

    def update(self, values: object, labels: object) -> None:
        label_values = np.asarray(labels, dtype=object)
        if label_values.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        matrix = _as_numeric_matrix(values, len(label_values), self.width)
        if not np.isfinite(matrix).all():
            raise ValueError("sufficient statistics require finite imputed values")
        unknown = sorted(set(label_values.astype(str)).difference(self.labels))
        if unknown:
            raise ValueError(f"labels are not active training classes: {unknown}")
        next_counts = self.counts.copy()
        next_means = self.means.copy()
        next_m2 = self.m2.copy()
        string_labels = label_values.astype(str)
        for label, class_index in self.label_to_index.items():
            selected = matrix[string_labels == label]
            if not len(selected):
                continue
            prospective = int(self.counts[class_index]) + len(selected)
            if prospective > np.iinfo(np.int64).max:
                raise OverflowError("class count exceeds int64")
            batch_mean, batch_m2 = _centered_batch_moments(selected)
            combined_mean, combined_m2 = _merge_centered_moments(
                int(self.counts[class_index]),
                self.means[class_index],
                self.m2[class_index],
                len(selected),
                batch_mean,
                batch_m2,
            )
            next_counts[class_index] = prospective
            next_means[class_index] = combined_mean
            next_m2[class_index] = combined_m2
        if not np.isfinite(next_means).all() or not np.isfinite(next_m2).all():
            raise OverflowError("updated sufficient statistics are non-finite")
        self.counts = next_counts
        self.means = next_means
        self.m2 = next_m2

    def merge(self, other: "ClassSufficientStatistics") -> "ClassSufficientStatistics":
        if not isinstance(other, ClassSufficientStatistics):
            raise TypeError("other must be ClassSufficientStatistics")
        if self.labels != other.labels or self.width != other.width:
            raise ValueError("incompatible sufficient statistics")
        prospective = [
            int(left) + int(right) for left, right in zip(self.counts, other.counts)
        ]
        if any(count > np.iinfo(np.int64).max for count in prospective):
            raise OverflowError("merged class count exceeds int64")
        next_counts = np.asarray(prospective, dtype=np.int64)
        next_means = self.means.copy()
        next_m2 = self.m2.copy()
        for index in range(len(self.labels)):
            next_means[index], next_m2[index] = _merge_centered_moments(
                int(self.counts[index]),
                self.means[index],
                self.m2[index],
                int(other.counts[index]),
                other.means[index],
                other.m2[index],
            )
        if not np.isfinite(next_means).all() or not np.isfinite(next_m2).all():
            raise OverflowError("merged sufficient statistics are non-finite")
        self.counts = next_counts
        self.means = next_means
        self.m2 = next_m2
        return self

    @property
    def total_rows(self) -> int:
        return int(sum(int(count) for count in self.counts))

    def population_variance(self) -> NDArray[np.float64]:
        total = self.total_rows
        if total == 0:
            return np.zeros(self.width, dtype=np.float64)
        _, total_m2 = self._combined_centered_moments()
        return np.maximum(total_m2 / total, 0.0)

    def _combined_centered_moments(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        count = 0
        mean: NDArray[np.float64] = np.zeros(self.width, dtype=np.float64)
        m2: NDArray[np.float64] = np.zeros(self.width, dtype=np.float64)
        for index in range(len(self.labels)):
            mean, m2 = _merge_centered_moments(
                count,
                mean,
                m2,
                int(self.counts[index]),
                self.means[index],
                self.m2[index],
            )
            count += int(self.counts[index])
        return mean, m2

    def finalize_f_scores(self) -> NDArray[np.float64]:
        total = self.total_rows
        class_count = len(self.labels)
        if total <= class_count or np.any(self.counts == 0):
            return np.zeros(self.width, dtype=np.float64)
        counts = self.counts.astype(np.float64)
        grand_mean, _ = self._combined_centered_moments()
        between = np.sum(
            counts[:, None] * np.square(self.means - grand_mean), axis=0
        )
        within = np.maximum(self.m2, 0.0).sum(axis=0)
        between_mean = between / (class_count - 1)
        within_mean = within / (total - class_count)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = between_mean / within_mean
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _centered_batch_moments(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    correction = centered.sum(axis=0, dtype=np.float64)
    m2 = np.square(centered).sum(axis=0, dtype=np.float64)
    m2 -= np.square(correction) / len(values)
    return mean, np.maximum(m2, 0.0)


def _merge_centered_moments(
    left_count: int,
    left_mean: NDArray[np.float64],
    left_m2: NDArray[np.float64],
    right_count: int,
    right_mean: NDArray[np.float64],
    right_m2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if right_count == 0:
        return left_mean.copy(), left_m2.copy()
    if left_count == 0:
        return right_mean.copy(), right_m2.copy()
    total = left_count + right_count
    delta = right_mean - left_mean
    mean = left_mean + delta * (right_count / total)
    m2 = left_m2 + right_m2
    m2 += np.square(delta) * (left_count * right_count / total)
    return mean, np.maximum(m2, 0.0)


class _MomentAccumulator:
    def __init__(self, width: int) -> None:
        self.width = width
        self.count = 0
        self.mean: NDArray[np.float64] = np.zeros(width, dtype=np.float64)
        self.m2: NDArray[np.float64] = np.zeros(width, dtype=np.float64)

    def update(self, values: NDArray[np.float64]) -> None:
        if values.ndim != 2 or values.shape[1] != self.width:
            raise ValueError("moment batch width mismatch")
        if not np.isfinite(values).all():
            raise ValueError("moments require finite imputed values")
        if not len(values):
            return
        if self.count + len(values) > np.iinfo(np.int64).max:
            raise OverflowError("moment count exceeds int64")
        batch_mean, batch_m2 = _centered_batch_moments(values)
        next_mean, next_m2 = _merge_centered_moments(
            self.count, self.mean, self.m2, len(values), batch_mean, batch_m2
        )
        if not np.isfinite(next_mean).all() or not np.isfinite(next_m2).all():
            raise OverflowError("updated moments are non-finite")
        self.count += len(values)
        self.mean = next_mean
        self.m2 = next_m2

    def merge(self, other: "_MomentAccumulator") -> "_MomentAccumulator":
        if not isinstance(other, _MomentAccumulator):
            raise TypeError("other must be _MomentAccumulator")
        if self.width != other.width:
            raise ValueError("incompatible moment accumulators")
        prospective = self.count + other.count
        if prospective > np.iinfo(np.int64).max:
            raise OverflowError("merged moment count exceeds int64")
        next_mean, next_m2 = _merge_centered_moments(
            self.count, self.mean, self.m2, other.count, other.mean, other.m2
        )
        if not np.isfinite(next_mean).all() or not np.isfinite(next_m2).all():
            raise OverflowError("merged moments are non-finite")
        self.count = prospective
        self.mean = next_mean
        self.m2 = next_m2
        return self

    def finalize(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.count <= 0:
            raise ValueError("moment accumulator is empty")
        return self.mean.copy(), np.maximum(self.m2 / self.count, 0.0)


class StreamingFeaturePreprocessor:
    """Three-pass train-only streaming preprocessor state machine."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        candidate_features: Sequence[str],
        split_fingerprint: str,
        expected_train_rows: int,
        quantile_capacity: int,
        quantile_seed: int,
    ) -> None:
        self.config = config
        self.original_candidate_features = tuple(str(name) for name in candidate_features)
        if not self.original_candidate_features or len(set(self.original_candidate_features)) != len(
            self.original_candidate_features
        ):
            raise ValueError("candidate_features must be non-empty and unique")
        if not isinstance(split_fingerprint, str) or not split_fingerprint:
            raise ValueError("split_fingerprint must be a non-empty string")
        if (
            isinstance(expected_train_rows, bool)
            or not isinstance(expected_train_rows, int)
            or expected_train_rows <= 0
        ):
            raise ValueError("expected_train_rows must be a positive integer")
        self.split_fingerprint = split_fingerprint
        self.expected_train_rows = expected_train_rows
        self.quantile_capacity = quantile_capacity
        self.quantile_seed = quantile_seed
        self.imputation_sketch = PriorityRowSketch(
            capacity=quantile_capacity,
            seed=quantile_seed,
            width=len(self.original_candidate_features),
        )
        self.selected_calibration: PriorityRowSketch | None = None
        self.benign_calibration: PriorityRowSketch | None = None
        self.anova: ClassSufficientStatistics | None = None
        self._selected_moments: _MomentAccumulator | None = None
        self._class_counts = {label: 0 for label in KNOWN_LABELS}
        self._state = "inspect"
        self._phase_rows = {phase: 0 for phase in (1, 2, 3)}
        self._usable_original_indices: NDArray[np.int64] | None = None
        self.candidate_features: list[str] = []
        self.medians: NDArray[np.float64] | None = None
        self.active_labels: list[str] = []
        self.selected_indices: NDArray[np.int64] | None = None
        self.selected_features: list[str] = []
        self.selection_scores: NDArray[np.float64] | None = None
        self.feature_costs: NDArray[np.float64] | None = None

        self._audit_root = Path(tempfile.mkdtemp(prefix="bitguard-preprocess-audit-"))
        self._audit_connection: sqlite3.Connection | None = sqlite3.connect(
            self._audit_root / "passes.sqlite3"
        )
        self._audit_connection.execute(
            "CREATE TABLE pass_rows ("
            "phase INTEGER NOT NULL, row_uid TEXT NOT NULL, label TEXT NOT NULL, "
            "row_digest TEXT NOT NULL, PRIMARY KEY (phase, row_uid))"
        )
        self._audit_connection.commit()

        # Fail early for invalid selection/scaler/encoder/cost configuration.
        FeaturePreprocessor(config)

    def _require_state(self, expected: str) -> None:
        if self._state != expected:
            raise RuntimeError(
                f"preprocessing state is {self._state}; expected {expected} phase"
            )

    def _validate_batch(
        self,
        row_uid: object,
        values: object,
        labels: object,
        *,
        split_fingerprint: str,
        feature_names: Sequence[str],
        membership: object,
    ) -> tuple[list[str], NDArray[np.float64], NDArray[np.str_]]:
        if split_fingerprint != self.split_fingerprint:
            raise ValueError("split fingerprint does not match the training plan")
        if tuple(str(name) for name in feature_names) != self.original_candidate_features:
            raise ValueError("feature tuple/order does not match the training plan")
        uids_raw = np.asarray(row_uid, dtype=object)
        labels_raw = np.asarray(labels, dtype=object)
        membership_raw = np.asarray(membership, dtype=object)
        if uids_raw.ndim != 1 or labels_raw.ndim != 1 or membership_raw.ndim != 1:
            raise ValueError("row_uid, labels, and membership must be one-dimensional")
        if not len(uids_raw):
            raise ValueError("train batch must not be empty")
        if len(labels_raw) != len(uids_raw) or len(membership_raw) != len(uids_raw):
            raise ValueError("train batch arrays must have equal row counts")
        if any(str(value) != "train" for value in membership_raw):
            raise ValueError("only effective split=train rows may fit preprocessing")
        uids = [str(value) for value in uids_raw]
        if any(not uid for uid in uids):
            raise ValueError("row_uid values must be non-empty strings")
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate row_uid within train batch")
        label_values = labels_raw.astype(str)
        if np.any(label_values == "unknown_like"):
            raise ValueError("unknown_like must not appear in streaming training")
        unsupported = sorted(set(label_values).difference(KNOWN_LABELS))
        if unsupported:
            raise ValueError(f"unsupported training behavior labels: {unsupported}")
        matrix = _as_numeric_matrix(
            values, len(uids), len(self.original_candidate_features)
        )
        return uids, matrix, label_values

    def _ensure_phase_uids_are_new(self, phase: int, uids: Sequence[str]) -> None:
        assert self._audit_connection is not None
        for uid in uids:
            duplicate = self._audit_connection.execute(
                "SELECT 1 FROM pass_rows WHERE phase = ? AND row_uid = ? LIMIT 1",
                (phase, uid),
            ).fetchone()
            if duplicate is not None:
                raise ValueError(f"duplicate row_uid in preprocessing pass {phase}: {uid}")

    def _record_audit(
        self,
        phase: int,
        uids: Sequence[str],
        labels: NDArray[np.str_],
        values: NDArray[np.float64],
    ) -> None:
        assert self._audit_connection is not None
        records = [
            (phase, uid, str(label), _row_value_digest(str(label), row))
            for uid, label, row in zip(uids, labels, values)
        ]
        try:
            with self._audit_connection:
                self._audit_connection.executemany(
                    "INSERT INTO pass_rows (phase, row_uid, label, row_digest) "
                    "VALUES (?, ?, ?, ?)",
                    records,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate row_uid in preprocessing pass {phase}") from exc
        self._phase_rows[phase] += len(uids)

    def _verify_phase(self, phase: int, *, compare_to_inspect: bool) -> str:
        if self._phase_rows[phase] != self.expected_train_rows:
            raise ValueError(
                f"preprocessing pass {phase} row count mismatch: "
                f"expected {self.expected_train_rows}, got {self._phase_rows[phase]}"
            )
        assert self._audit_connection is not None
        if compare_to_inspect:
            mismatch = self._audit_connection.execute(
                "SELECT 1 FROM ("
                "SELECT row_uid, label, row_digest FROM pass_rows WHERE phase = 1 "
                "EXCEPT SELECT row_uid, label, row_digest FROM pass_rows WHERE phase = ? "
                "UNION ALL "
                "SELECT row_uid, label, row_digest FROM pass_rows WHERE phase = ? "
                "EXCEPT SELECT row_uid, label, row_digest FROM pass_rows WHERE phase = 1"
                ") LIMIT 1",
                (phase, phase),
            ).fetchone()
            if mismatch is not None:
                raise ValueError(f"preprocessing pass {phase} mismatch: training rows changed")
        digest = hashlib.sha256()
        cursor = self._audit_connection.execute(
            "SELECT row_uid, label, row_digest FROM pass_rows "
            "WHERE phase = ? ORDER BY row_uid",
            (phase,),
        )
        for uid, label, row_digest in cursor:
            digest.update(str(uid).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(label).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(row_digest).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def inspect_batch(
        self,
        row_uid: object,
        values: object,
        labels: object,
        *,
        split_fingerprint: str,
        feature_names: Sequence[str],
        membership: object,
    ) -> None:
        self._require_state("inspect")
        uids, matrix, label_values = self._validate_batch(
            row_uid,
            values,
            labels,
            split_fingerprint=split_fingerprint,
            feature_names=feature_names,
            membership=membership,
        )
        self._ensure_phase_uids_are_new(_PHASE_INSPECT, uids)
        normalized = matrix.astype(np.float32).astype(np.float64)
        self.imputation_sketch.update_many(uids, normalized)
        for label in label_values:
            self._class_counts[str(label)] += 1
        self._record_audit(_PHASE_INSPECT, uids, label_values, matrix)

    def finalize_imputation(self) -> None:
        self._require_state("inspect")
        self._inspect_uid_digest = self._verify_phase(
            _PHASE_INSPECT, compare_to_inspect=False
        )
        self.active_labels = [
            label for label in KNOWN_LABELS if self._class_counts[label] > 0
        ]
        if len(self.active_labels) < 2:
            raise ValueError("streaming training needs at least two known behavior classes")
        finite_counts = self.imputation_sketch.finite_counts
        usable: NDArray[np.int64] = np.flatnonzero(finite_counts > 0).astype(
            np.int64
        )
        if not len(usable):
            raise ValueError("all candidate features are entirely missing in training")
        medians: list[float] = []
        for original_index in usable:
            feature = self.original_candidate_features[int(original_index)]
            try:
                medians.append(
                    self.imputation_sketch.quantile(int(original_index), 0.5)
                )
            except ValueError as exc:
                raise ValueError(
                    f"no finite retained sample for usable feature {feature}; "
                    "increase quantile sketch capacity"
                ) from exc
        self._usable_original_indices = usable
        self.candidate_features = [
            self.original_candidate_features[int(index)] for index in usable
        ]
        self.medians = np.asarray(medians, dtype=np.float64)
        self.anova = ClassSufficientStatistics(
            self.active_labels, len(self.candidate_features)
        )
        self._state = "anova"

    def _impute_usable(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._usable_original_indices is not None and self.medians is not None
        usable = matrix[:, self._usable_original_indices].astype(np.float32).astype(
            np.float64
        )
        medians = self.medians.astype(np.float32).astype(np.float64)
        return np.where(np.isfinite(usable), usable, medians)

    def accumulate_anova_batch(
        self,
        row_uid: object,
        values: object,
        labels: object,
        *,
        split_fingerprint: str,
        feature_names: Sequence[str],
        membership: object,
    ) -> None:
        self._require_state("anova")
        uids, matrix, label_values = self._validate_batch(
            row_uid,
            values,
            labels,
            split_fingerprint=split_fingerprint,
            feature_names=feature_names,
            membership=membership,
        )
        self._ensure_phase_uids_are_new(_PHASE_ANOVA, uids)
        assert self.anova is not None
        self.anova.update(self._impute_usable(matrix), label_values)
        self._record_audit(_PHASE_ANOVA, uids, label_values, matrix)

    def finalize_selection(self) -> None:
        self._require_state("anova")
        self._verify_phase(_PHASE_ANOVA, compare_to_inspect=True)
        assert self.anova is not None
        scores = self.anova.finalize_f_scores()
        reference = FeaturePreprocessor(self.config)
        costs: NDArray[np.float64] = reference._cost_lookup(
            self.candidate_features
        ).astype(np.float64)
        selection = str(self.config["preprocess"].get("selection", "f_score"))
        if selection == "variance":
            rank_score = self.anova.population_variance()
        elif selection == "cost_aware":
            normalized = scores / max(float(scores.max()), 1e-12)
            rank_score = normalized / costs
        elif selection in {"f_score", "expert"}:
            rank_score = scores
        else:
            raise ValueError(f"unsupported feature selection: {selection}")
        feature_budget = self.config["preprocess"].get("feature_budget")
        budget = len(self.candidate_features) if feature_budget is None else int(feature_budget)
        budget = min(max(budget, 1), len(self.candidate_features))
        if selection == "expert":
            expert = [
                str(item)
                for item in self.config["preprocess"].get("expert_features", [])
            ]
            if not expert:
                raise ValueError("selection=expert requires preprocess.expert_features")
            missing = [feature for feature in expert if feature not in self.candidate_features]
            if missing:
                raise ValueError(f"expert features missing from dataset: {missing}")
            selected = np.asarray(
                [self.candidate_features.index(feature) for feature in expert[:budget]],
                dtype=np.int64,
            )
        else:
            selected = np.argsort(-rank_score, kind="stable")[:budget].astype(np.int64)
        self.selected_indices = selected
        self.selected_features = [self.candidate_features[int(index)] for index in selected]
        self.selection_scores = rank_score[selected].astype(np.float64)
        self.feature_costs = costs[selected].astype(np.float64)
        self.selected_calibration = PriorityRowSketch(
            capacity=self.quantile_capacity,
            seed=self.quantile_seed,
            width=len(selected),
        )
        self.benign_calibration = PriorityRowSketch(
            capacity=self.quantile_capacity,
            seed=self.quantile_seed,
            width=len(selected),
        )
        self._selected_moments = _MomentAccumulator(len(selected))
        self._state = "calibration"

    def calibrate_selected_batch(
        self,
        row_uid: object,
        values: object,
        labels: object,
        *,
        split_fingerprint: str,
        feature_names: Sequence[str],
        membership: object,
    ) -> None:
        self._require_state("calibration")
        uids, matrix, label_values = self._validate_batch(
            row_uid,
            values,
            labels,
            split_fingerprint=split_fingerprint,
            feature_names=feature_names,
            membership=membership,
        )
        self._ensure_phase_uids_are_new(_PHASE_CALIBRATE, uids)
        assert (
            self.selected_indices is not None
            and self.selected_calibration is not None
            and self.benign_calibration is not None
            and self._selected_moments is not None
        )
        selected = self._impute_usable(matrix)[:, self.selected_indices]
        self.selected_calibration.update_many(uids, selected)
        self._selected_moments.update(selected)
        benign_mask = label_values == "benign"
        if np.any(benign_mask):
            benign_uids = [uid for uid, keep in zip(uids, benign_mask) if keep]
            self.benign_calibration.update_many(benign_uids, selected[benign_mask])
        self._record_audit(_PHASE_CALIBRATE, uids, label_values, matrix)

    def _hydrate_scaler(self, result: FeaturePreprocessor) -> None:
        assert self.selected_calibration is not None and self._selected_moments is not None
        width = len(self.selected_features)
        scaler_name = str(self.config["preprocess"].get("scaler", "robust"))
        if scaler_name == "standard":
            mean, variance = self._selected_moments.finalize()
            epsilon = np.finfo(np.float64).eps
            upper_bound = (
                self._selected_moments.count * epsilon * variance
                + np.square(self._selected_moments.count * mean * epsilon)
            )
            scale = np.sqrt(variance)
            scale[variance <= upper_bound] = 1.0
            result.scaler.mean_ = mean
            result.scaler.var_ = variance
            result.scaler.scale_ = scale
            result.scaler.n_samples_seen_ = np.int64(self._selected_moments.count)
            result.scaler.n_features_in_ = width
        elif scaler_name == "robust":
            result.scaler.fit(self._retained_matrix(self.selected_calibration))
        elif scaler_name == "none":
            result.scaler.n_features_in_ = width
        else:  # Constructor validation should make this unreachable.
            raise ValueError(f"unsupported scaler: {scaler_name}")

    @staticmethod
    def _retained_matrix(sketch: PriorityRowSketch) -> NDArray[np.float32]:
        retained = sketch.retained_rows()
        if not retained:
            raise ValueError("calibration sketch has no retained rows")
        return np.stack([values for _, values in retained]).astype(np.float32)

    def _hydrate_encoder(self, result: FeaturePreprocessor) -> None:
        assert self.selected_calibration is not None
        retained = self._retained_matrix(self.selected_calibration)
        scaled = result.scaler.transform(retained).astype(np.float32)
        result.encoder.fit(scaled)

    def _hydrate_open_set(self, result: FeaturePreprocessor) -> None:
        assert self.benign_calibration is not None
        if self.benign_calibration.total_rows == 0:
            raise ValueError("open-set calibration requires benign training rows")
        retained = self._retained_matrix(self.benign_calibration)
        scaled = result.scaler.transform(retained).astype(np.float32)
        center = np.median(scaled, axis=0).astype(np.float32)
        distances = np.mean(np.abs(scaled - center), axis=1)
        quantile = float(
            self.config["preprocess"]["open_set"].get(
                "benign_distance_quantile", 0.99
            )
        )
        if not 0.5 <= quantile < 1.0:
            raise ValueError("benign_distance_quantile must be in [0.5, 1.0)")
        result.benign_center = center
        result.open_distance_threshold = max(
            float(np.quantile(distances, quantile)),
            1e-6,
        )

    @staticmethod
    def _sketch_provenance(sketch: PriorityRowSketch) -> dict[str, Any]:
        return {
            "total_rows": sketch.total_rows,
            "retained_rows": sketch.retained_count,
            "exact": sketch.retained_count == sketch.total_rows,
            "confidence": sketch.confidence_metadata(confidence=0.95),
        }

    def finalize(self) -> FeaturePreprocessor:
        self._require_state("calibration")
        self._verify_phase(_PHASE_CALIBRATE, compare_to_inspect=True)
        assert (
            self.medians is not None
            and self.selected_indices is not None
            and self.selection_scores is not None
            and self.feature_costs is not None
            and self.selected_calibration is not None
            and self.benign_calibration is not None
        )
        result = FeaturePreprocessor(self.config)
        result.candidate_features = list(self.candidate_features)
        result.selected_indices = self.selected_indices.copy()
        result.selected_features = list(self.selected_features)
        result.selection_scores = self.selection_scores.astype(np.float32)
        result.feature_costs = self.feature_costs.astype(np.float32)
        result.active_labels = list(self.active_labels)
        result.label_to_index = {
            label: index for index, label in enumerate(result.active_labels)
        }
        result.imputer.statistics_ = self.medians.astype(np.float32)
        result.imputer.n_features_in_ = len(self.candidate_features)
        result.imputer._fit_dtype = np.dtype(np.float32)
        result.imputer._fill_dtype = np.dtype(np.float32)
        result.imputer.indicator_ = None
        self._hydrate_scaler(result)
        self._hydrate_encoder(result)
        self._hydrate_open_set(result)
        sketches = {
            "imputation": self._sketch_provenance(self.imputation_sketch),
            "selected": self._sketch_provenance(self.selected_calibration),
            "benign": self._sketch_provenance(self.benign_calibration),
        }
        exact_fields = [
            "train_membership_and_pass_identity",
            "finite_missing_counts",
            "class_counts",
            "anova_sufficient_statistics_after_frozen_imputation",
            "feature_costs_and_stable_ranking",
        ]
        approximate_fields: list[str] = []
        quantile_fields = [
            ("imputation_medians", "imputation"),
            ("benign_center_and_distance_quantile", "benign"),
        ]
        scaler_name = str(self.config["preprocess"].get("scaler", "robust"))
        if scaler_name == "standard":
            exact_fields.append("standard_scaler_moments")
        elif scaler_name == "robust":
            quantile_fields.append(("robust_scaler_quantiles", "selected"))
        if result.encoder.kind != "none":
            quantile_fields.append(("encoder_quantile_thresholds", "selected"))
        for field, sketch_name in quantile_fields:
            target = exact_fields if sketches[sketch_name]["exact"] else approximate_fields
            target.append(field)
        missing_seen = bool(np.any(self.imputation_sketch.missing_counts > 0))
        result.fit_provenance = {
            "fit_mode": "streaming_priority_sketch",
            "version": STREAMING_PREPROCESS_VERSION,
            "passes": 3,
            "split_fingerprint": self.split_fingerprint,
            "train_uid_digest": self._inspect_uid_digest,
            "rows_considered": self.expected_train_rows,
            "candidate_feature_order": list(self.original_candidate_features),
            "quantile_algorithm": self.imputation_sketch.algorithm,
            "quantile_algorithm_version": self.imputation_sketch.version,
            "quantile_capacity": self.quantile_capacity,
            "quantile_seed": self.quantile_seed,
            "quantile_confidence": sketches["imputation"]["confidence"],
            "benign_quantile_confidence": sketches["benign"]["confidence"],
            "sketches": sketches,
            "retained_counts": {
                "imputation": self.imputation_sketch.retained_count,
                "selected": self.selected_calibration.retained_count,
                "benign": self.benign_calibration.retained_count,
            },
            "exact_fields": exact_fields,
            "approximate_fields": approximate_fields,
            "anova_imputation_semantics": (
                "ANOVA is exact for the frozen imputed values; medians are priority-sketch "
                "approximations when missing values are present."
                if missing_seen
                else "ANOVA is exact; no imputation replacement was required."
            ),
            "validation_calibration_used": False,
        }
        result.fitted = True
        self._state = "finalized"
        self._close_audit()
        return result

    def _close_audit(self) -> None:
        if self._audit_connection is not None:
            self._audit_connection.close()
            self._audit_connection = None
        if self._audit_root.exists():
            shutil.rmtree(self._audit_root)

    def __del__(self) -> None:
        try:
            self._close_audit()
        except Exception:
            pass


__all__ = [
    "ClassSufficientStatistics",
    "STREAMING_PREPROCESS_VERSION",
    "StreamingFeaturePreprocessor",
]
