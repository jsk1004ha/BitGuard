from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


_CHECK_ORDER = (
    "macro_f1",
    "balanced_accuracy",
    "required_class_support",
    "required_class_recall",
    "high_risk_recall",
    "fixed_fpr_attack_recall",
    "fixed_fpr_benign_fpr",
    "ece",
    "active_groups",
)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _is_unit_interval(value: float) -> bool:
    return 0.0 <= value <= 1.0


def _label_list(policy: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = policy.get(name)
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"deployment policy {name} must contain labels")
    labels = tuple(value)
    if len(set(labels)) != len(labels):
        raise ValueError(f"deployment policy {name} must not contain duplicates")
    return labels


def _policy_probability(policy: Mapping[str, object], name: str) -> float:
    value = _finite_number(policy.get(name))
    if value is None or not 0.0 <= value <= 1.0:
        raise ValueError(f"deployment policy {name} must be finite in [0, 1]")
    return value


def _per_class_value(
    per_class: object, label: str, field: str
) -> float | None:
    if not isinstance(per_class, Mapping):
        return None
    values = per_class.get(label)
    if not isinstance(values, Mapping):
        return None
    return _finite_number(values.get(field))


def assess_deployment_candidate(
    metrics: Mapping[str, object],
    model_summary: Mapping[str, object],
    policy: Mapping[str, object],
) -> dict[str, Any]:
    """Assess a completed test set without changing any training decision.

    The test-set result is an approval gate only. It must never feed thresholds or
    hyperparameters back into the fitted model.
    """

    if not isinstance(metrics, Mapping) or not isinstance(model_summary, Mapping):
        raise TypeError("metrics and model_summary must be mappings")
    if not isinstance(policy, Mapping):
        raise TypeError("deployment policy must be a mapping")

    required_classes = _label_list(policy, "required_classes")
    high_risk_classes = _label_list(policy, "high_risk_classes")
    minimum_support = policy.get("min_required_class_support")
    minimum_active = policy.get("min_active_groups")
    if (
        isinstance(minimum_support, bool)
        or not isinstance(minimum_support, Integral)
        or int(minimum_support) <= 0
    ):
        raise ValueError("min_required_class_support must be a positive integer")
    if (
        isinstance(minimum_active, bool)
        or not isinstance(minimum_active, Integral)
        or int(minimum_active) < 0
    ):
        raise ValueError("min_active_groups must be a non-negative integer")

    minimums = {
        "macro_f1": _policy_probability(policy, "min_macro_f1"),
        "balanced_accuracy": _policy_probability(
            policy, "min_balanced_accuracy"
        ),
        "required_class_recall": _policy_probability(
            policy, "min_required_class_recall"
        ),
        "high_risk_recall": _policy_probability(
            policy, "min_high_risk_recall"
        ),
        "fixed_fpr_attack_recall": _policy_probability(
            policy, "min_attack_recall_at_fixed_fpr"
        ),
    }
    maxima = {
        "fixed_fpr_benign_fpr": _policy_probability(
            policy, "max_observed_benign_fpr"
        ),
        "ece": _policy_probability(policy, "max_ece"),
    }
    fixed_fpr_target = _policy_probability(policy, "fixed_fpr_target")
    if fixed_fpr_target <= 0.0:
        raise ValueError("fixed_fpr_target must be positive")

    checks: dict[str, dict[str, object]] = {}

    def minimum_check(name: str, actual: float | None) -> None:
        expected = minimums[name]
        checks[name] = {
            "passed": actual is not None
            and _is_unit_interval(actual)
            and actual >= expected,
            "actual": actual,
            "minimum": expected,
        }

    def maximum_check(name: str, actual: float | None) -> None:
        expected = maxima[name]
        checks[name] = {
            "passed": actual is not None
            and _is_unit_interval(actual)
            and actual <= expected,
            "actual": actual,
            "maximum": expected,
        }

    minimum_check("macro_f1", _finite_number(metrics.get("macro_f1")))
    minimum_check(
        "balanced_accuracy", _finite_number(metrics.get("balanced_accuracy"))
    )

    per_class = metrics.get("per_class")
    supports = {
        label: _per_class_value(per_class, label, "support")
        for label in required_classes
    }
    recalls = {
        label: _per_class_value(per_class, label, "recall")
        for label in required_classes
    }
    checks["required_class_support"] = {
        "passed": all(
            value is not None
            and value >= 0.0
            and value.is_integer()
            and value >= int(minimum_support)
            for value in supports.values()
        ),
        "actual": supports,
        "minimum": int(minimum_support),
    }
    checks["required_class_recall"] = {
        "passed": all(
            value is not None
            and _is_unit_interval(value)
            and value >= minimums["required_class_recall"]
            for value in recalls.values()
        ),
        "actual": recalls,
        "minimum": minimums["required_class_recall"],
    }
    high_risk_recalls = {
        label: _per_class_value(per_class, label, "recall")
        for label in high_risk_classes
    }
    checks["high_risk_recall"] = {
        "passed": bool(high_risk_recalls)
        and all(
            value is not None
            and _is_unit_interval(value)
            and value >= minimums["high_risk_recall"]
            for value in high_risk_recalls.values()
        ),
        "actual": high_risk_recalls,
        "minimum": minimums["high_risk_recall"],
    }

    fixed_fpr = metrics.get("fixed_fpr")
    target_name = f"{fixed_fpr_target:g}"
    attack_recall = (
        _finite_number(
            fixed_fpr.get(f"attack_recall_at_benign_fpr_{target_name}")
        )
        if isinstance(fixed_fpr, Mapping)
        else None
    )
    observed_benign_fpr = (
        _finite_number(
            fixed_fpr.get(f"observed_benign_fpr_at_target_{target_name}")
        )
        if isinstance(fixed_fpr, Mapping)
        else None
    )
    minimum_check("fixed_fpr_attack_recall", attack_recall)
    maximum_check("fixed_fpr_benign_fpr", observed_benign_fpr)
    maximum_check(
        "ece", _finite_number(metrics.get("expected_calibration_error_10_bin"))
    )

    active_groups = model_summary.get("active_groups")
    normalized_active = (
        int(active_groups)
        if isinstance(active_groups, Integral) and not isinstance(active_groups, bool)
        else None
    )
    checks["active_groups"] = {
        "passed": normalized_active is not None
        and normalized_active >= 0
        and normalized_active >= int(minimum_active),
        "actual": normalized_active,
        "minimum": int(minimum_active),
    }

    failed_checks = [
        name for name in _CHECK_ORDER if not bool(checks[name]["passed"])
    ]
    return {
        "format_version": 1,
        "eligible": not failed_checks,
        "status": "deployment_candidate" if not failed_checks else "rejected",
        "failed_checks": failed_checks,
        "checks": checks,
        "policy": dict(policy),
    }


__all__ = ["assess_deployment_candidate"]
