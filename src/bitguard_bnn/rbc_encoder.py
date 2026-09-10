from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RawComparison:
    """One deployable raw-space comparison bit."""

    feature: str
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "threshold": self.threshold, "operator": ">="}


class RiskWeightedComparisonEncoder:
    """Fixed-budget binary encoder selected by benign/attack collision risk.

    ``fit`` expects training data only. It creates an internal search/selection
    split: quantile thresholds are generated on the search partition and greedy
    replacements are accepted only when they reduce collision cost on the
    selection partition. Transform output uses ``{-1, +1}``, matching BitGuard's
    binary first-layer convention.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        bit_budget: int,
        *,
        benign_label: str | int | bool = "benign",
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 5.0,
        selection_fraction: float = 0.25,
        candidate_quantiles: Sequence[float] | None = None,
        raw_feature_cost_cap: float | None = None,
        random_state: int = 2309,
        max_replacements: int | None = None,
        max_swaps: int | None = None,
        min_comparison_support: int = 2,
        min_improvement: float = 1e-12,
    ) -> None:
        if int(bit_budget) != bit_budget or bit_budget <= 0:
            raise ValueError("bit_budget must be a positive integer")
        if false_positive_cost <= 0 or false_negative_cost <= 0:
            raise ValueError("collision costs must be positive")
        if not 0.0 < selection_fraction < 1.0:
            raise ValueError("selection_fraction must be between 0 and 1")
        if raw_feature_cost_cap is not None and raw_feature_cost_cap <= 0:
            raise ValueError("raw_feature_cost_cap must be positive")
        if max_replacements is not None and max_swaps is not None:
            raise ValueError("use only one of max_replacements and max_swaps")
        resolved_max_replacements = max_replacements if max_swaps is None else max_swaps
        if resolved_max_replacements is not None and resolved_max_replacements < 0:
            raise ValueError("max_replacements must be non-negative")
        if min_comparison_support < 1:
            raise ValueError("min_comparison_support must be positive")

        quantiles = (
            np.linspace(0.05, 0.95, 19, dtype=np.float64)
            if candidate_quantiles is None
            else np.asarray(candidate_quantiles, dtype=np.float64)
        )
        if quantiles.ndim != 1 or len(quantiles) == 0:
            raise ValueError("candidate_quantiles must be a non-empty sequence")
        if not np.isfinite(quantiles).all() or np.any((quantiles <= 0) | (quantiles >= 1)):
            raise ValueError("candidate_quantiles must be finite and strictly between 0 and 1")

        self.bit_budget = int(bit_budget)
        self.benign_label = benign_label
        self.false_positive_cost = float(false_positive_cost)
        self.false_negative_cost = float(false_negative_cost)
        self.selection_fraction = float(selection_fraction)
        self.candidate_quantiles = tuple(float(value) for value in quantiles)
        self.raw_feature_cost_cap = (
            None if raw_feature_cost_cap is None else float(raw_feature_cost_cap)
        )
        self.random_state = int(random_state)
        self.max_replacements = resolved_max_replacements
        self.min_comparison_support = int(min_comparison_support)
        self.min_improvement = float(min_improvement)

        self.feature_names_: list[str] = []
        self.imputation_values_: dict[str, float] = {}
        self.feature_costs_: dict[str, float] = {}
        self.comparisons_: list[RawComparison] = []
        self.initial_comparisons_: list[RawComparison] = []
        self.initial_selection_collision_: float | None = None
        self.selection_collision_: float | None = None
        self.replacement_history_: list[dict[str, Any]] = []
        self.search_row_count_: int = 0
        self.selection_row_count_: int = 0
        self.group_disjoint_selection_: bool = False
        self.fitted_: bool = False

    def fit(
        self,
        frame: pd.DataFrame,
        labels: Sequence[Any] | pd.Series | np.ndarray,
        *,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        attack_groups: Sequence[Any] | pd.Series | np.ndarray | None = None,
        groups: Sequence[Any] | pd.Series | np.ndarray | None = None,
        feature_costs: Mapping[str, float] | None = None,
    ) -> "RiskWeightedComparisonEncoder":
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.columns.duplicated().any():
            raise ValueError("frame columns must be unique")
        numeric = frame.select_dtypes(include=[np.number]).columns.astype(str).tolist()
        if not numeric:
            raise ValueError("frame has no numeric feature columns")

        y_raw = np.asarray(labels)
        if y_raw.ndim != 1 or len(y_raw) != len(frame):
            raise ValueError("labels must be one-dimensional and match frame rows")
        attack = self._attack_mask(y_raw)
        if not attack.any() or attack.all():
            raise ValueError("labels must contain both benign and attack rows")

        weights = self._risk_weights(attack, sample_weight, attack_groups)
        search_indices, selection_indices = self._internal_split(attack, groups)
        self.search_row_count_ = int(len(search_indices))
        self.selection_row_count_ = int(len(selection_indices))
        self.feature_names_ = numeric
        self.feature_costs_ = self._validated_feature_costs(numeric, feature_costs)

        raw = frame.loc[:, numeric].replace([np.inf, -np.inf], np.nan).to_numpy(np.float64)
        search_raw = raw[search_indices]
        medians = np.nanmedian(search_raw, axis=0)
        if np.isnan(medians).any():
            missing = [numeric[index] for index in np.flatnonzero(np.isnan(medians))]
            raise ValueError(f"numeric features entirely missing in search partition: {missing}")
        self.imputation_values_ = {
            feature: float(medians[index]) for index, feature in enumerate(numeric)
        }
        values = np.where(np.isnan(raw), medians[None, :], raw)
        search_values = values[search_indices]
        selection_values = values[selection_indices]
        selection_attack = attack[selection_indices]
        selection_weights = weights[selection_indices]

        candidates = self._candidate_comparisons(search_values)
        comparisons = self._initial_comparisons(search_values)
        if len(comparisons) != self.bit_budget:
            raise RuntimeError("failed to construct the requested bit budget")
        self._check_cost_cap(comparisons)
        self.initial_comparisons_ = list(comparisons)

        cost = self._collision_for_comparisons(
            selection_values, selection_attack, selection_weights, comparisons
        )
        self.initial_selection_collision_ = cost
        self.replacement_history_ = []
        limit = self.bit_budget if self.max_replacements is None else self.max_replacements

        for _ in range(limit):
            best: tuple[float, int, RawComparison] | None = None
            best_key: tuple[float, int, str, float] | None = None
            current_keys = {(item.feature, item.threshold) for item in comparisons}
            for position in range(self.bit_budget):
                removed = comparisons[position]
                for candidate in candidates:
                    key = (candidate.feature, candidate.threshold)
                    if candidate == removed or key in current_keys:
                        continue
                    trial = list(comparisons)
                    trial[position] = candidate
                    if not self._within_cost_cap(trial):
                        continue
                    trial_cost = self._collision_for_comparisons(
                        selection_values, selection_attack, selection_weights, trial
                    )
                    proposal_key = (trial_cost, position, candidate.feature, candidate.threshold)
                    if best_key is None or proposal_key < best_key:
                        best = (trial_cost, position, candidate)
                        best_key = proposal_key
            if best is None or best[0] >= cost - self.min_improvement:
                break
            new_cost, position, candidate = best
            removed = comparisons[position]
            comparisons[position] = candidate
            self.replacement_history_.append(
                {
                    "bit": position,
                    "removed": removed.to_dict(),
                    "added": candidate.to_dict(),
                    "collision_before": cost,
                    "collision_after": new_cost,
                }
            )
            cost = new_cost

        self.comparisons_ = comparisons
        self.selection_collision_ = cost
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        missing = [feature for feature in self.feature_names_ if feature not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing fitted numeric features: {missing}")
        raw = (
            frame.loc[:, self.feature_names_]
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(np.float64)
        )
        medians = np.asarray(
            [self.imputation_values_[feature] for feature in self.feature_names_],
            dtype=np.float64,
        )
        values = np.where(np.isnan(raw), medians[None, :], raw)
        bits = self._boolean_codes(values, self.comparisons_)
        return np.where(bits, 1.0, -1.0).astype(np.float32)

    def fit_transform(
        self,
        frame: pd.DataFrame,
        labels: Sequence[Any] | pd.Series | np.ndarray,
        **fit_kwargs: Any,
    ) -> np.ndarray:
        return self.fit(frame, labels, **fit_kwargs).transform(frame)

    def collision_cost(
        self,
        frame: pd.DataFrame,
        labels: Sequence[Any] | pd.Series | np.ndarray,
        *,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        attack_groups: Sequence[Any] | pd.Series | np.ndarray | None = None,
    ) -> float:
        """Return the empirical weighted ambiguity of the fitted binary codes."""

        self._check_fitted()
        y_raw = np.asarray(labels)
        if y_raw.ndim != 1 or len(y_raw) != len(frame):
            raise ValueError("labels must be one-dimensional and match frame rows")
        attack = self._attack_mask(y_raw)
        weights = self._risk_weights(attack, sample_weight, attack_groups)
        codes = self.transform(frame) > 0
        return self._collision_cost(codes, attack, weights)

    @property
    def encoded_dimension(self) -> int:
        return self.bit_budget

    @property
    def raw_feature_cost(self) -> float:
        self._check_fitted()
        used = {comparison.feature for comparison in self.comparisons_}
        return float(sum(self.feature_costs_[feature] for feature in used))

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "format_version": self.FORMAT_VERSION,
            "type": type(self).__name__,
            "config": {
                "bit_budget": self.bit_budget,
                "benign_label": self.benign_label,
                "false_positive_cost": self.false_positive_cost,
                "false_negative_cost": self.false_negative_cost,
                "selection_fraction": self.selection_fraction,
                "candidate_quantiles": list(self.candidate_quantiles),
                "raw_feature_cost_cap": self.raw_feature_cost_cap,
                "random_state": self.random_state,
                "max_replacements": self.max_replacements,
                "min_comparison_support": self.min_comparison_support,
                "min_improvement": self.min_improvement,
            },
            "feature_names": list(self.feature_names_),
            "imputation_values": dict(self.imputation_values_),
            "feature_costs": dict(self.feature_costs_),
            "comparisons": [comparison.to_dict() for comparison in self.comparisons_],
            "initial_comparisons": [
                comparison.to_dict() for comparison in self.initial_comparisons_
            ],
            "selection": {
                "search_rows": self.search_row_count_,
                "selection_rows": self.selection_row_count_,
                "group_disjoint": self.group_disjoint_selection_,
                "initial_collision": self.initial_selection_collision_,
                "final_collision": self.selection_collision_,
                "replacement_history": list(self.replacement_history_),
            },
            "raw_feature_cost": self.raw_feature_cost,
        }

    @classmethod
    def from_dict(cls, artifact: Mapping[str, Any]) -> "RiskWeightedComparisonEncoder":
        if artifact.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported RBC encoder format version")
        config = dict(artifact["config"])
        encoder = cls(**config)
        encoder.feature_names_ = [str(item) for item in artifact["feature_names"]]
        encoder.imputation_values_ = {
            str(key): float(value) for key, value in artifact["imputation_values"].items()
        }
        encoder.feature_costs_ = {
            str(key): float(value) for key, value in artifact["feature_costs"].items()
        }
        encoder.comparisons_ = cls._comparisons_from_artifact(artifact["comparisons"])
        encoder.initial_comparisons_ = cls._comparisons_from_artifact(
            artifact["initial_comparisons"]
        )
        selection = artifact["selection"]
        encoder.search_row_count_ = int(selection["search_rows"])
        encoder.selection_row_count_ = int(selection["selection_rows"])
        encoder.group_disjoint_selection_ = bool(selection.get("group_disjoint", False))
        encoder.initial_selection_collision_ = float(selection["initial_collision"])
        encoder.selection_collision_ = float(selection["final_collision"])
        encoder.replacement_history_ = list(selection["replacement_history"])
        if len(encoder.comparisons_) != encoder.bit_budget:
            raise ValueError("serialized comparison count does not match bit budget")
        if set(encoder.imputation_values_) != set(encoder.feature_names_):
            raise ValueError("serialized imputation values do not match feature names")
        encoder._check_cost_cap(encoder.comparisons_)
        encoder.fitted_ = True
        return encoder

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RiskWeightedComparisonEncoder":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            raise TypeError("RBC encoder artifact must contain a JSON object")
        return cls.from_dict(artifact)

    def _internal_split(
        self,
        attack: np.ndarray,
        groups: Sequence[Any] | pd.Series | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import GroupShuffleSplit, train_test_split

        indices = np.arange(len(attack), dtype=np.int64)
        class_counts = np.bincount(attack.astype(np.int8), minlength=2)
        if len(attack) < 4 or np.any(class_counts < 2):
            raise ValueError("internal selection needs at least two benign and two attack rows")
        if groups is not None:
            group_values = np.asarray(groups)
            if group_values.ndim != 1 or len(group_values) != len(attack):
                raise ValueError("groups must be one-dimensional and match frame rows")
            if len(np.unique(group_values)) < 2:
                raise ValueError("group-disjoint selection needs at least two groups")
            for offset in range(32):
                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=self.selection_fraction,
                    random_state=self.random_state + offset,
                )
                search, selection = next(splitter.split(indices, attack, group_values))
                if np.unique(attack[search]).size == 2 and np.unique(attack[selection]).size == 2:
                    self.group_disjoint_selection_ = True
                    return np.sort(search), np.sort(selection)
            raise ValueError(
                "could not create a group-disjoint selection split containing both classes"
            )
        self.group_disjoint_selection_ = False
        search, selection = train_test_split(
            indices,
            test_size=self.selection_fraction,
            random_state=self.random_state,
            shuffle=True,
            stratify=attack,
        )
        return np.sort(search), np.sort(selection)

    def _validated_feature_costs(
        self, features: list[str], feature_costs: Mapping[str, float] | None
    ) -> dict[str, float]:
        provided = {} if feature_costs is None else dict(feature_costs)
        unknown = sorted(set(provided) - set(features))
        if unknown:
            raise ValueError(f"feature_costs contains unknown numeric features: {unknown}")
        result = {feature: float(provided.get(feature, 1.0)) for feature in features}
        if any(not np.isfinite(value) or value <= 0 for value in result.values()):
            raise ValueError("feature costs must be finite and positive")
        return result

    def _initial_comparisons(self, search_values: np.ndarray) -> list[RawComparison]:
        eligible = [
            feature
            for feature in self.feature_names_
            if self.raw_feature_cost_cap is None
            or self.feature_costs_[feature] <= self.raw_feature_cost_cap + 1e-12
        ]
        if not eligible:
            raise ValueError("raw_feature_cost_cap cannot fund any numeric feature")

        chosen: list[str] = []
        running_cost = 0.0
        for feature in eligible:
            cost = self.feature_costs_[feature]
            if self.raw_feature_cost_cap is None or running_cost + cost <= self.raw_feature_cost_cap + 1e-12:
                chosen.append(feature)
                running_cost += cost
            if len(chosen) == self.bit_budget:
                break
        if not chosen:
            raise ValueError("raw_feature_cost_cap cannot fund an initial comparison")

        allocation = {feature: 0 for feature in chosen}
        for bit in range(self.bit_budget):
            allocation[chosen[bit % len(chosen)]] += 1
        comparisons: list[RawComparison] = []
        feature_index = {feature: index for index, feature in enumerate(self.feature_names_)}
        for feature in chosen:
            count = allocation[feature]
            quantiles = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
            thresholds = np.quantile(search_values[:, feature_index[feature]], quantiles)
            comparisons.extend(
                RawComparison(feature, float(threshold)) for threshold in np.atleast_1d(thresholds)
            )
        return comparisons

    def _candidate_comparisons(self, search_values: np.ndarray) -> list[RawComparison]:
        result: list[RawComparison] = []
        for index, feature in enumerate(self.feature_names_):
            thresholds = np.unique(
                np.quantile(search_values[:, index], np.asarray(self.candidate_quantiles))
            )
            for value in thresholds:
                left = int(np.count_nonzero(search_values[:, index] < value))
                right = int(len(search_values) - left)
                if min(left, right) >= self.min_comparison_support:
                    result.append(RawComparison(feature, float(value)))
        return result

    def _risk_weights(
        self,
        attack: np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None,
        attack_groups: Sequence[Any] | pd.Series | np.ndarray | None,
    ) -> np.ndarray:
        if sample_weight is None:
            weights = np.ones(len(attack), dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.ndim != 1 or len(weights) != len(attack):
                raise ValueError("sample_weight must be one-dimensional and match frame rows")
            if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("sample_weight must be finite, non-negative, and non-zero")
        if attack_groups is not None:
            groups = np.asarray(attack_groups)
            if groups.ndim != 1 or len(groups) != len(attack):
                raise ValueError("attack_groups must be one-dimensional and match frame rows")
            attack_group_values = groups[attack]
            for group in np.unique(attack_group_values):
                mask = attack & (groups == group)
                total = weights[mask].sum()
                if total > 0:
                    weights[mask] /= total
        return weights

    def _attack_mask(self, labels: np.ndarray) -> np.ndarray:
        if np.issubdtype(labels.dtype, np.bool_):
            return labels.astype(bool, copy=False)
        if np.issubdtype(labels.dtype, np.number):
            unique = np.unique(labels)
            if np.isin(unique, [0, 1]).all():
                return labels.astype(bool)
        return np.asarray(labels != self.benign_label, dtype=bool)

    def _boolean_codes(
        self, values: np.ndarray, comparisons: Sequence[RawComparison]
    ) -> np.ndarray:
        feature_index = {feature: index for index, feature in enumerate(self.feature_names_)}
        return np.column_stack(
            [
                values[:, feature_index[item.feature]] >= item.threshold
                for item in comparisons
            ]
        )

    def _collision_for_comparisons(
        self,
        values: np.ndarray,
        attack: np.ndarray,
        weights: np.ndarray,
        comparisons: Sequence[RawComparison],
    ) -> float:
        return self._collision_cost(
            self._boolean_codes(values, comparisons), attack, weights
        )

    def _collision_cost(
        self, codes: np.ndarray, attack: np.ndarray, weights: np.ndarray
    ) -> float:
        packed = np.packbits(codes, axis=1, bitorder="little")
        _, inverse = np.unique(packed, axis=0, return_inverse=True)
        benign_weight = np.bincount(
            inverse, weights=weights * (~attack), minlength=int(inverse.max()) + 1
        )
        attack_weight = np.bincount(
            inverse, weights=weights * attack, minlength=int(inverse.max()) + 1
        )
        return float(
            np.minimum(
                self.false_positive_cost * benign_weight,
                self.false_negative_cost * attack_weight,
            ).sum()
        )

    def _within_cost_cap(self, comparisons: Sequence[RawComparison]) -> bool:
        if self.raw_feature_cost_cap is None:
            return True
        used = {comparison.feature for comparison in comparisons}
        cost = sum(self.feature_costs_[feature] for feature in used)
        return cost <= self.raw_feature_cost_cap + 1e-12

    def _check_cost_cap(self, comparisons: Sequence[RawComparison]) -> None:
        if not self._within_cost_cap(comparisons):
            raise ValueError("comparisons exceed raw_feature_cost_cap")

    @staticmethod
    def _comparisons_from_artifact(items: Sequence[Mapping[str, Any]]) -> list[RawComparison]:
        result: list[RawComparison] = []
        for item in items:
            if item.get("operator") != ">=":
                raise ValueError("unsupported serialized comparison operator")
            threshold = float(item["threshold"])
            if not np.isfinite(threshold):
                raise ValueError("serialized thresholds must be finite")
            result.append(RawComparison(str(item["feature"]), threshold))
        return result

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("RBC encoder is not fitted")


RBCEncoder = RiskWeightedComparisonEncoder
