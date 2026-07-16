from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import ACTION_NAMES


@dataclass
class SecurityCounters:
    scan: int = 0
    flood: int = 0
    beacon: int = 0
    unknown: int = 0
    benign: int = 0
    last_timestamp: float | None = None

    def clip(self) -> None:
        for name in ("scan", "flood", "beacon", "unknown", "benign"):
            setattr(self, name, int(np.clip(getattr(self, name), 0, 15)))


class TemporalSecurityStateMachine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config["temporal"] if "temporal" in config else config
        self.states: OrderedDict[str, SecurityCounters] = OrderedDict()
        self.max_devices = int(self.config.get("max_devices", 4096))
        self.evictions = 0
        self.increment = int(self.config.get("increment", 2))
        self.decay = int(self.config.get("decay", 1))
        self.evidence_threshold = float(self.config.get("evidence_threshold", 0.45))
        self.decay_interval_seconds = float(self.config.get("decay_interval_seconds", 60.0))
        self.risk_weights = dict(self.config["risk_weights"])
        self.action_thresholds = np.asarray(self.config["action_thresholds"], dtype=np.float64)

    def reset(self, device_id: str | None = None) -> None:
        if device_id is None:
            self.states.clear()
        else:
            self.states.pop(str(device_id), None)

    def get(self, device_id: object) -> SecurityCounters:
        key = str(device_id)
        if key in self.states:
            self.states.move_to_end(key)
            return self.states[key]
        if len(self.states) >= self.max_devices:
            self.states.popitem(last=False)
            self.evictions += 1
        self.states[key] = SecurityCounters()
        return self.states[key]

    def temporal_suspicion(self, device_id: object) -> float:
        state = self.get(device_id)
        suspicious = state.scan + state.flood + state.beacon + state.unknown
        return float(np.clip(suspicious / 60.0 - state.benign / 60.0, 0.0, 1.0))

    def _time_decay(self, state: SecurityCounters, timestamp: float | None) -> int:
        if timestamp is None or not np.isfinite(timestamp):
            return 0
        if state.last_timestamp is None:
            state.last_timestamp = float(timestamp)
            return 0
        elapsed = max(float(timestamp) - state.last_timestamp, 0.0)
        intervals = int(elapsed // max(self.decay_interval_seconds, 1e-6))
        if intervals > 0:
            state.last_timestamp += intervals * self.decay_interval_seconds
        return intervals * self.decay

    def update(
        self,
        device_id: object,
        probabilities: dict[str, float],
        timestamp: float | None = None,
    ) -> tuple[SecurityCounters, float, int]:
        state = self.get(device_id)
        elapsed_decay = self._time_decay(state, timestamp)
        evidence = {
            "scan": float(probabilities.get("scan_like", 0.0)),
            "flood": float(probabilities.get("flood_like", 0.0)),
            "beacon": float(probabilities.get("beacon_like", 0.0)),
            "unknown": float(probabilities.get("unknown_like", 0.0)),
            "benign": float(probabilities.get("benign", 0.0)),
        }
        for name, score in evidence.items():
            delta = self.increment if score >= self.evidence_threshold else -self.decay
            setattr(state, name, getattr(state, name) + delta - elapsed_decay)
        state.clip()
        attack_probability = 1.0 - float(probabilities.get("benign", 0.0))
        risk = self.risk(device_id, attack_probability)
        action = int(np.searchsorted(self.action_thresholds, risk, side="right"))
        return state, risk, action

    def risk(self, device_id: object, attack_probability: float) -> float:
        state = self.get(device_id)
        weights = self.risk_weights
        risk = (
            float(weights["model"]) * attack_probability
            + float(weights["scan"]) * state.scan / 15.0
            + float(weights["flood"]) * state.flood / 15.0
            + float(weights["beacon"]) * state.beacon / 15.0
            + float(weights["unknown"]) * state.unknown / 15.0
            - float(weights["benign"]) * state.benign / 15.0
        )
        return float(np.clip(risk, 0.0, 1.0))


def temporal_state_key(source_file: object, device_id: object) -> str:
    """Encode an episode/device tuple without delimiter ambiguity."""

    return json.dumps(
        [str(source_file), str(device_id)],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def replay_predictions(predictions: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"device_id", "true_label", "predicted_label"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction replay missing columns: {sorted(missing)}")
    probability_columns = [column for column in predictions if column.startswith("prob_")]
    if not probability_columns:
        raise ValueError("prediction replay needs prob_<label> columns")
    order_columns = [column for column in ("timestamp", "sequence_index") if column in predictions]
    replay = predictions.copy()
    replay["__original_order"] = np.arange(len(replay))
    if "timestamp" in replay and replay["timestamp"].notna().all():
        replay = replay.sort_values(["timestamp", "device_id", "__original_order"], kind="stable")
        has_real_time = bool(replay.get("has_wall_clock_time", pd.Series(False)).all())
    elif "sequence_index" in replay:
        replay = replay.sort_values(["device_id", "sequence_index", "__original_order"], kind="stable")
        has_real_time = False
    else:
        has_real_time = False
    machine = TemporalSecurityStateMachine(config)
    output_rows: list[dict[str, Any]] = []
    for row in replay.to_dict(orient="records"):
        output_rows.append(
            replay_prediction_row(row, machine, probability_columns)
        )
    result_frame = pd.DataFrame(output_rows).drop(columns=["__original_order"], errors="ignore")
    continuous = bool(result_frame.get("temporal_continuity", pd.Series(False)).all())
    metrics = operational_metrics(result_frame, has_real_time and continuous, continuous)
    metrics["temporal_state_device_evictions"] = machine.evictions
    metrics["temporal_state_peak_device_capacity"] = machine.max_devices
    return result_frame, metrics


def replay_prediction_row(
    row: dict[str, Any],
    machine: TemporalSecurityStateMachine,
    probability_columns: Sequence[str],
) -> dict[str, Any]:
    """Apply one ordered prediction while preserving temporal machine state."""

    probabilities = {
        column.removeprefix("prob_"): float(row[column])
        for column in probability_columns
    }
    timestamp = row.get("timestamp")
    timestamp_value = (
        float(timestamp) if timestamp is not None and pd.notna(timestamp) else None
    )
    episode = str(row.get("source_file", "default_episode"))
    state_key = temporal_state_key(episode, row["device_id"])
    state, risk, action = machine.update(state_key, probabilities, timestamp_value)
    result = dict(row)
    result.update(
        {
            f"state_{key}": value
            for key, value in asdict(state).items()
            if key != "last_timestamp"
        }
    )
    result["risk_score"] = risk
    result["action_level"] = action
    result["action"] = ACTION_NAMES[action]
    if action < 2:
        result["stateful_predicted_label"] = "benign"
    elif row["predicted_label"] != "benign":
        result["stateful_predicted_label"] = row["predicted_label"]
    else:
        dominant = max(
            ("scan", "flood", "beacon", "unknown"),
            key=lambda name: getattr(state, name),
        )
        result["stateful_predicted_label"] = f"{dominant}_like"
    return result


def operational_metrics(
    frame: pd.DataFrame, has_real_time: bool, continuous: bool = True
) -> dict[str, Any]:
    benign = frame["true_label"] == "benign"
    attack = ~benign
    alerts = frame["action_level"] >= 2
    mitigations = frame["action_level"] >= 3
    metrics: dict[str, Any] = {
        "rows": len(frame),
        "alerts": int(alerts.sum()),
        "mitigation_recommendations": int(mitigations.sum()),
        "benign_disruption_rate": float(mitigations[benign].mean()) if benign.any() else None,
        "action_recommendation_precision": float(attack[mitigations].mean()) if mitigations.any() else None,
        "attack_row_recall_at_alert": float(alerts[attack].mean()) if attack.any() else None,
        "attack_row_recall_at_mitigation": float(mitigations[attack].mean()) if attack.any() else None,
        "alert_reduction_ratio_vs_alert_every_row": float(1.0 - alerts.mean()),
        "packed_counter_state_bytes_per_device_theoretical": 3,
        "counter_count": 5,
        "automatic_blocking_performed": False,
        "attack_reduction_sensitivity": {
            "25_percent_effective_after_level3": float(mitigations[attack].mean() * 0.25) if attack.any() else None,
            "50_percent_effective_after_level3": float(mitigations[attack].mean() * 0.50) if attack.any() else None,
            "75_percent_effective_after_level3": float(mitigations[attack].mean() * 0.75) if attack.any() else None,
        },
    }
    stateless_alerts = frame["predicted_label"] != "benign"
    metrics["stateless_alerts"] = int(stateless_alerts.sum())
    metrics["alert_reduction_ratio_vs_stateless_classifier"] = (
        float(1.0 - alerts.sum() / stateless_alerts.sum()) if stateless_alerts.any() else None
    )
    metrics["stateless_attack_recall"] = (
        float(stateless_alerts[attack].mean()) if attack.any() else None
    )
    metrics["stateful_attack_recall_at_level2"] = (
        float(alerts[attack].mean()) if attack.any() else None
    )
    low_rate = frame.get("raw_attack", pd.Series("", index=frame.index)).astype(str).str.contains(
        "low|slow|beacon", case=False, regex=True
    )
    metrics["low_rate_recall_at_level2"] = float(alerts[low_rate].mean()) if low_rate.any() else None
    delays_decisions: list[float] = []
    delays_seconds: list[float] = []
    mitigation_delays: list[float] = []
    attack_episode_count = 0
    missed_alert_episodes = 0
    missed_mitigation_episodes = 0
    false_positive_alert_events = 0
    episode_columns = [column for column in ("source_file", "device_id") if column in frame]
    for _, device in frame.groupby(episode_columns, sort=False):
        attack_values = device["true_label"].to_numpy() != "benign"
        action_values = device["action_level"].to_numpy()
        alert_values = action_values >= 2
        alert_rising = alert_values & ~np.r_[False, alert_values[:-1]]
        false_positive_alert_events += int(np.sum(alert_rising & ~attack_values))
        starts = np.flatnonzero(attack_values & ~np.r_[False, attack_values[:-1]])
        ends = np.flatnonzero(~attack_values & np.r_[False, attack_values[:-1]])
        if attack_values.any() and attack_values[-1]:
            ends = np.r_[ends, len(device)]
        for onset, end in zip(starts, ends):
            attack_episode_count += 1
            alert_positions = np.flatnonzero(action_values[onset:end] >= 2)
            if len(alert_positions):
                detected = int(onset + alert_positions[0])
                delays_decisions.append(float(detected - onset))
                if has_real_time:
                    delays_seconds.append(
                        float(device.iloc[detected]["timestamp"] - device.iloc[onset]["timestamp"])
                    )
            else:
                missed_alert_episodes += 1
            mitigation_positions = np.flatnonzero(action_values[onset:end] >= 3)
            if len(mitigation_positions):
                detected = int(onset + mitigation_positions[0])
                mitigation_delays.append(
                    float(device.iloc[detected]["timestamp"] - device.iloc[onset]["timestamp"])
                    if has_real_time
                    else float(detected - onset)
                )
            else:
                missed_mitigation_episodes += 1
    metrics["detection_delay_decisions_p50"] = (
        float(np.percentile(delays_decisions, 50)) if delays_decisions and continuous else None
    )
    metrics["detection_delay_decisions_p95"] = (
        float(np.percentile(delays_decisions, 95)) if delays_decisions and continuous else None
    )
    metrics["detection_delay_seconds_p50"] = (
        float(np.percentile(delays_seconds, 50)) if delays_seconds and has_real_time else None
    )
    metrics["detection_delay_seconds_p95"] = (
        float(np.percentile(delays_seconds, 95)) if delays_seconds and has_real_time else None
    )
    metrics["time_to_mitigation_p50_seconds_or_decisions"] = (
        float(np.percentile(mitigation_delays, 50)) if mitigation_delays and continuous else None
    )
    metrics["time_to_mitigation_unit"] = "seconds" if has_real_time else "decisions"
    metrics["attack_episode_count"] = attack_episode_count
    metrics["missed_alert_episode_count"] = missed_alert_episodes
    metrics["missed_mitigation_episode_count"] = missed_mitigation_episodes
    metrics["delay_note"] = "Delay percentiles are conditional on detection; misses are reported separately."
    metrics["false_positive_alert_events"] = false_positive_alert_events
    if has_real_time:
        hours = 0.0
        for _, device in frame.groupby(episode_columns):
            duration = float(device["timestamp"].max() - device["timestamp"].min()) / 3600.0
            hours += max(duration, 1.0 / 3600.0)
        metrics["false_positive_alerts_per_device_hour"] = float(
            false_positive_alert_events / hours
        )
        metrics["observed_device_hours"] = hours
    else:
        metrics["false_positive_alerts_per_device_hour"] = None
        metrics["observed_device_hours"] = None
        metrics["time_metric_note"] = (
            "No verified continuous wall-clock episode; device-hour and time-delay metrics are withheld."
        )
    metrics["temporal_continuity_verified"] = continuous
    return metrics
