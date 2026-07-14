from __future__ import annotations

import re
from typing import Final

KNOWN_LABELS: Final[list[str]] = [
    "benign",
    "scan_like",
    "flood_like",
    "beacon_like",
    "exfil_like",
]
UNKNOWN_LABEL: Final[str] = "unknown_like"
CANONICAL_LABELS: Final[list[str]] = [*KNOWN_LABELS, UNKNOWN_LABEL]

META_COLUMNS: Final[set[str]] = {
    "row_uid",
    "dataset",
    "source_file",
    "sequence_index",
    "device_id",
    "raw_attack",
    "behavior_label",
    "timestamp",
}

COMMON_STREAM_FEATURES: Final[list[str]] = [
    "packet_rate",
    "byte_rate",
    "burst_score",
    "tcp_ratio",
    "udp_ratio",
    "icmp_ratio",
    "syn_ratio",
    "rst_ratio",
    "ack_only_ratio",
    "unique_destination_ip_ratio",
    "unique_destination_port_ratio",
    "new_destination_ratio",
    "repeated_destination_score",
    "repeated_port_score",
    "interarrival_mean",
    "interarrival_jitter",
    "interarrival_stability",
    "periodicity_score",
    "outbound_packet_ratio",
    "outbound_byte_ratio",
    "short_flow_ratio",
    "long_flow_ratio",
    "small_packet_ratio",
    "failed_connection_score",
]

ACTION_NAMES: Final[list[str]] = [
    "allow",
    "log_only",
    "monitor_alert",
    "rate_limit_recommendation",
    "temporary_isolation_recommendation",
    "administrator_confirmed_quarantine_recommendation",
]


def normalize_token(value: object) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return token or "unknown"


def nbaiot_behavior(raw_attack: object) -> str:
    """Map N-BaIoT family/subtype names to defensible behavior proxies.

    N-BaIoT does not contain labelled beacon or exfiltration behavior. Flooding
    subtypes therefore remain flood-like; only explicit scans become scan-like.
    """

    raw = normalize_token(raw_attack)
    if raw in {"benign", "normal", "benign_traffic"}:
        return "benign"
    if raw.endswith("_scan") or raw == "scan":
        return "scan_like"
    return "flood_like"


def botiot_behavior(category: object, subcategory: object | None = None) -> str:
    token = normalize_token(f"{category}_{subcategory or ''}")
    if any(part in token for part in ("normal", "benign")):
        return "benign"
    if any(part in token for part in ("scan", "reconnaissance", "fingerprint")):
        return "scan_like"
    if any(part in token for part in ("ddos", "dos", "flood")):
        return "flood_like"
    if any(part in token for part in ("exfil", "theft", "keylog", "data_exfiltration")):
        return "exfil_like"
    if any(part in token for part in ("beacon", "c2", "command_control")):
        return "beacon_like"
    return UNKNOWN_LABEL


def canonicalize_behavior(value: object) -> str:
    token = normalize_token(value)
    aliases = {
        "normal": "benign",
        "attack": "unknown_like",
        "scan": "scan_like",
        "flood": "flood_like",
        "beacon": "beacon_like",
        "exfil": "exfil_like",
        "unknown": "unknown_like",
    }
    token = aliases.get(token, token)
    if token not in CANONICAL_LABELS:
        return UNKNOWN_LABEL
    return token


def infer_feature_cost(name: str) -> float:
    """Return a normalized *proxy* cost, not a hardware energy measurement."""

    token = normalize_token(name)
    if any(x in token for x in ("destination", "unique", "cardinality", "sketch")):
        return 1.00
    if any(x in token for x in ("period", "jitter", "interarrival", "repeated")):
        return 0.80
    if any(x in token for x in ("mean", "std", "variance", "covariance", "radius")):
        return 0.60
    if any(x in token for x in ("byte", "packet", "count", "rate", "ratio", "flag")):
        return 0.30
    return 0.50

