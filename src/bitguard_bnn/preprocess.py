from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import resolve_path
from .constants import KNOWN_LABELS, infer_feature_cost


class IdentityScaler:
    def fit(self, values: np.ndarray) -> "IdentityScaler":
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)


@dataclass
class BinaryFeatureEncoder:
    kind: str
    bits: int
    thresholds: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "BinaryFeatureEncoder":
        values = np.asarray(values, dtype=np.float32)
        if self.kind == "none":
            self.thresholds = None
        elif self.kind == "sign":
            self.thresholds = np.nanmedian(values, axis=0).astype(np.float32)
        elif self.kind == "thermometer":
            quantiles = np.arange(1, self.bits + 1, dtype=np.float32) / (self.bits + 1)
            self.thresholds = np.quantile(values, quantiles, axis=0).T.astype(np.float32)
        elif self.kind == "hybrid":
            levels = 2**self.bits
            quantiles = np.arange(1, levels, dtype=np.float32) / levels
            self.thresholds = np.quantile(values, quantiles, axis=0).T.astype(np.float32)
        else:
            raise ValueError(f"unsupported encoder: {self.kind}")
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.kind == "none":
            return values
        if self.thresholds is None:
            raise RuntimeError("encoder is not fitted")
        if self.kind == "sign":
            return np.where(values >= self.thresholds, 1.0, -1.0).astype(np.float32)
        comparisons = values[:, :, None] >= self.thresholds[None, :, :]
        if self.kind == "thermometer":
            return np.where(comparisons, 1.0, -1.0).reshape(len(values), -1).astype(np.float32)
        bins = comparisons.sum(axis=2).astype(np.float32)
        max_bin = float(self.thresholds.shape[1])
        return (2.0 * bins / max_bin - 1.0).astype(np.float32)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def encoded_dimension(self, original_dimension: int) -> int:
        return original_dimension * self.bits if self.kind == "thermometer" else original_dimension

    def input_groups(self, original_dimension: int) -> np.ndarray:
        if self.kind == "thermometer":
            return np.repeat(np.arange(original_dimension, dtype=np.int64), self.bits)
        return np.arange(original_dimension, dtype=np.int64)

    def encoded_indices_for_first(self, count: int, original_dimension: int) -> np.ndarray:
        count = min(max(int(count), 1), original_dimension)
        groups = self.input_groups(original_dimension)
        return np.flatnonzero(groups < count)


class FeaturePreprocessor:
    """Train-only preprocessing, feature ranking, encoding, and open-set calibration."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        cfg = config["preprocess"]
        self.feature_budget = cfg.get("feature_budget")
        self.selection = str(cfg.get("selection", "f_score"))
        self.imputer = SimpleImputer(strategy="median")
        scaler_name = str(cfg.get("scaler", "robust"))
        if scaler_name == "robust":
            self.scaler: Any = RobustScaler(quantile_range=(25.0, 75.0))
        elif scaler_name == "standard":
            self.scaler = StandardScaler()
        elif scaler_name == "none":
            self.scaler = IdentityScaler()
        else:
            raise ValueError(f"unsupported scaler: {scaler_name}")
        self.encoder = BinaryFeatureEncoder(
            str(cfg.get("encoder", "sign")), int(cfg.get("thermometer_bits", 2))
        )
        self.candidate_features: list[str] = []
        self.selected_features: list[str] = []
        self.selected_indices: np.ndarray | None = None
        self.selection_scores: np.ndarray | None = None
        self.feature_costs: np.ndarray | None = None
        self.active_labels: list[str] = []
        self.label_to_index: dict[str, int] = {}
        self.benign_center: np.ndarray | None = None
        self.open_distance_threshold: float = 1.0
        self.fit_provenance: dict[str, Any] = {
            "fit_mode": "in_memory_exact",
            "passes": 1,
            "exact_fields": [
                "imputation_medians",
                "feature_selection",
                "scaler_statistics",
                "encoder_thresholds",
                "benign_center_and_distance_quantile",
            ],
            "approximate_fields": [],
            "validation_calibration_used": False,
        }
        self.fitted = False

    @property
    def fit_provenance(self) -> dict[str, Any]:
        provenance = getattr(
            self,
            "_fit_provenance",
            {"fit_mode": "in_memory_exact", "validation_calibration_used": False},
        )
        return copy.deepcopy(provenance)

    @fit_provenance.setter
    def fit_provenance(self, value: dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise TypeError("fit_provenance must be a dictionary")
        self._fit_provenance = copy.deepcopy(value)

    def _cost_lookup(self, features: list[str]) -> np.ndarray:
        path_value = self.config["preprocess"].get("feature_cost_csv")
        explicit: dict[str, float] = {}
        if path_value:
            path = resolve_path(self.config, path_value)
            assert path is not None
            costs = pd.read_csv(path)
            if not {"feature", "cost"}.issubset(costs.columns):
                raise ValueError("feature cost CSV must contain feature,cost columns")
            explicit = dict(zip(costs["feature"].astype(str), costs["cost"].astype(float)))
        values = np.asarray(
            [explicit.get(feature, infer_feature_cost(feature)) for feature in features],
            dtype=np.float32,
        )
        if np.any(values <= 0) or not np.isfinite(values).all():
            raise ValueError("feature costs must be finite and positive")
        return values / values.mean()

    def fit(self, train: pd.DataFrame, candidate_features: list[str]) -> "FeaturePreprocessor":
        self.fit_provenance = {
            "fit_mode": "in_memory_exact",
            "passes": 1,
            "exact_fields": [
                "imputation_medians",
                "feature_selection",
                "scaler_statistics",
                "encoder_thresholds",
                "benign_center_and_distance_quantile",
            ],
            "approximate_fields": [],
            "validation_calibration_used": False,
        }
        if (train["behavior_label"] == "unknown_like").any():
            raise ValueError("unknown_like must not appear in training; use an attack-held-out test")
        self.candidate_features = list(candidate_features)
        if not self.candidate_features:
            raise ValueError("candidate feature list is empty")
        raw_frame = train[self.candidate_features].replace([np.inf, -np.inf], np.nan)
        usable = raw_frame.notna().any(axis=0).to_numpy()
        if not usable.all():
            self.candidate_features = [
                feature for feature, keep in zip(self.candidate_features, usable) if keep
            ]
            raw_frame = raw_frame.loc[:, self.candidate_features]
        if not self.candidate_features:
            raise ValueError("all candidate features are entirely missing in training")
        raw = raw_frame.to_numpy(np.float32)
        imputed = self.imputer.fit_transform(raw).astype(np.float32)
        y_names = train["behavior_label"].astype(str).to_numpy()
        self.active_labels = [label for label in KNOWN_LABELS if label in set(y_names)]
        if len(self.active_labels) < 2:
            raise ValueError("training needs at least two known behavior classes")
        self.label_to_index = {label: index for index, label in enumerate(self.active_labels)}
        y = np.asarray([self.label_to_index[label] for label in y_names], dtype=np.int64)
        try:
            scores, _ = f_classif(imputed, y)
        except ValueError:
            scores = np.nanvar(imputed, axis=0)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        costs: np.ndarray = self._cost_lookup(self.candidate_features).astype(np.float64)
        if self.selection == "variance":
            rank_score = np.nanvar(imputed, axis=0)
        elif self.selection == "cost_aware":
            normalized = scores / max(float(scores.max()), 1e-12)
            rank_score = normalized / costs
        elif self.selection in {"f_score", "expert"}:
            rank_score = scores
        else:
            raise ValueError(f"unsupported feature selection: {self.selection}")
        budget = len(self.candidate_features) if self.feature_budget is None else int(self.feature_budget)
        budget = min(max(budget, 1), len(self.candidate_features))
        if self.selection == "expert":
            expert = [str(item) for item in self.config["preprocess"].get("expert_features", [])]
            if not expert:
                raise ValueError("selection=expert requires preprocess.expert_features")
            missing = [feature for feature in expert if feature not in self.candidate_features]
            if missing:
                raise ValueError(f"expert features missing from dataset: {missing}")
            selected = np.asarray(
                [self.candidate_features.index(feature) for feature in expert[:budget]], dtype=np.int64
            )
        else:
            selected = np.argsort(-rank_score, kind="stable")[:budget]
        self.selected_indices = selected.astype(np.int64)
        self.selected_features = [self.candidate_features[index] for index in selected]
        self.selection_scores = np.asarray(rank_score[selected], dtype=np.float32)
        self.feature_costs = np.asarray(costs[selected], dtype=np.float32)
        selected_imputed = imputed[:, selected]
        scaled = self.scaler.fit(selected_imputed).transform(selected_imputed).astype(np.float32)
        self.encoder.fit(scaled)
        benign = scaled[y_names == "benign"]
        if len(benign) == 0:
            raise ValueError("open-set calibration requires benign training rows")
        self.benign_center = np.median(benign, axis=0).astype(np.float32)
        benign_distance = np.mean(np.abs(benign - self.benign_center), axis=1)
        quantile = float(
            self.config["preprocess"]["open_set"].get("benign_distance_quantile", 0.99)
        )
        if not 0.5 <= quantile < 1.0:
            raise ValueError("benign_distance_quantile must be in [0.5, 1.0)")
        self.open_distance_threshold = max(float(np.quantile(benign_distance, quantile)), 1e-6)
        self.fitted = True
        return self

    def transform_unencoded(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        raw = frame[self.candidate_features].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
        imputed = self.imputer.transform(raw).astype(np.float32)
        assert self.selected_indices is not None
        return self.scaler.transform(imputed[:, self.selected_indices]).astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.encoder.transform(self.transform_unencoded(frame))

    def calibrate_open_set(self, validation: pd.DataFrame) -> "FeaturePreprocessor":
        """Calibrate benign-distance threshold on a disjoint validation partition."""

        self._check_fitted()
        benign_mask = validation["behavior_label"].astype(str).to_numpy() == "benign"
        if not benign_mask.any():
            return self
        values = self.transform_unencoded(validation)[benign_mask]
        distance = self.anomaly_distance(values)
        quantile = float(
            self.config["preprocess"]["open_set"].get("benign_distance_quantile", 0.99)
        )
        self.open_distance_threshold = max(float(np.quantile(distance, quantile)), 1e-6)
        return self

    def encode_labels(self, frame: pd.DataFrame, unknown_value: int = -1) -> np.ndarray:
        self._check_fitted()
        return np.asarray(
            [self.label_to_index.get(str(label), unknown_value) for label in frame["behavior_label"]],
            dtype=np.int64,
        )

    def anomaly_distance(self, unencoded_values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.benign_center is not None
        return np.mean(np.abs(unencoded_values - self.benign_center), axis=1)

    def apply_open_set(
        self, probabilities: np.ndarray, unencoded_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predicted labels, unknown scores, and six-column probabilities."""

        self._check_fitted()
        probabilities = np.asarray(probabilities, dtype=np.float64)
        max_probability = probabilities.max(axis=1)
        known_index = probabilities.argmax(axis=1)
        known_label = np.asarray([self.active_labels[index] for index in known_index], dtype=object)
        distance = self.anomaly_distance(unencoded_values)
        confidence_threshold = float(
            self.config["preprocess"]["open_set"].get("confidence_threshold", 0.60)
        )
        distance_ratio = distance / self.open_distance_threshold
        anomaly_strength = 1.0 / (1.0 + np.exp(-4.0 * (distance_ratio - 1.0)))
        uncertainty = (
            np.zeros_like(max_probability)
            if confidence_threshold <= 0.0
            else np.clip(
                (confidence_threshold - max_probability) / confidence_threshold,
                0.0,
                1.0,
            )
        )
        unknown_score = np.clip(anomaly_strength * uncertainty, 0.0, 0.999)
        enabled = bool(self.config["preprocess"]["open_set"].get("enabled", True))
        is_unknown = enabled & (max_probability < confidence_threshold) & (distance >= self.open_distance_threshold)
        labels = known_label.copy()
        labels[is_unknown] = "unknown_like"
        expanded = probabilities * (1.0 - unknown_score[:, None])
        expanded = np.column_stack([expanded, unknown_score])
        expanded /= np.maximum(expanded.sum(axis=1, keepdims=True), 1e-12)
        return labels.astype(str), unknown_score.astype(np.float32), expanded.astype(np.float32)

    def calibrate_confidence_threshold(
        self,
        known_probabilities: np.ndarray,
        unencoded_values: np.ndarray,
    ) -> float:
        """Tune confidence on known validation rows without using held-out attacks."""

        distance = self.anomaly_distance(unencoded_values)
        maximum = np.asarray(known_probabilities).max(axis=1)
        anomaly = distance >= self.open_distance_threshold
        target = float(
            self.config["preprocess"]["open_set"].get(
                "max_known_false_unknown_rate", 0.02
            )
        )
        if not 0.0 <= target < 0.5:
            raise ValueError("max_known_false_unknown_rate must be in [0, 0.5)")
        chosen = 0.0
        for threshold in np.linspace(0.05, 0.99, 189):
            false_unknown_rate = float(np.mean(anomaly & (maximum < threshold)))
            if false_unknown_rate <= target + 1e-12:
                chosen = float(threshold)
            else:
                break
        self.config["preprocess"]["open_set"]["confidence_threshold"] = chosen
        return chosen

    @property
    def encoded_dimension(self) -> int:
        self._check_fitted()
        return self.encoder.encoded_dimension(len(self.selected_features))

    @property
    def input_groups(self) -> np.ndarray:
        self._check_fitted()
        return self.encoder.input_groups(len(self.selected_features))

    def feature_manifest(self) -> dict[str, Any]:
        self._check_fitted()
        assert self.feature_costs is not None and self.selection_scores is not None
        provenance = self.fit_provenance
        return {
            "fit_mode": str(provenance.get("fit_mode", "in_memory_exact")),
            "fit_provenance": provenance,
            "candidate_count": len(self.candidate_features),
            "selected_count": len(self.selected_features),
            "selected_features": self.selected_features,
            "selection_method": self.selection,
            "selection_scores": self.selection_scores.tolist(),
            "normalized_proxy_costs": self.feature_costs.tolist(),
            "cost_warning": "Proxy costs are normalized design assumptions, not measured energy.",
            "encoder": self.encoder.kind,
            "encoder_bits": self.encoder.bits,
            "encoded_dimension": self.encoded_dimension,
            "active_supervised_labels": self.active_labels,
            "unknown_strategy": "low known-class confidence AND robust benign-distance anomaly",
            "benign_distance_threshold": self.open_distance_threshold,
            "open_set_confidence_threshold": float(
                self.config["preprocess"]["open_set"]["confidence_threshold"]
            ),
        }

    def save(self, path: Path) -> None:
        self._check_fitted()
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "FeaturePreprocessor":
        result = joblib.load(path)
        if not isinstance(result, FeaturePreprocessor):
            raise TypeError("artifact is not a FeaturePreprocessor")
        return result

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("preprocessor is not fitted")


def class_weights(labels: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("all active classes must have training examples")
    weights = len(labels) / (class_count * counts)
    return (weights / weights.mean()).astype(np.float32)
