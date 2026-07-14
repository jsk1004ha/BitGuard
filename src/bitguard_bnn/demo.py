from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import COMMON_STREAM_FEATURES


def generate_demo(path: Path, rows: int = 12_000, seed: int = 2309) -> pd.DataFrame:
    if rows < 1_000:
        raise ValueError("demo needs at least 1,000 rows")
    rng = np.random.default_rng(seed)
    devices = np.asarray([f"device_{index:02d}" for index in range(8)])
    behavior = rng.choice(
        ["benign", "scan_like", "flood_like", "beacon_like", "exfil_like", "unknown_like"],
        size=rows,
        p=[0.67, 0.09, 0.11, 0.05, 0.05, 0.03],
    )
    raw_map = {
        "benign": "benign",
        "scan_like": "port_scan",
        "flood_like": "udp_flood",
        "beacon_like": "periodic_c2",
        "exfil_like": "data_exfiltration",
        "unknown_like": "novel_lowrate",
    }
    frame = pd.DataFrame(
        {
            "device_id": devices[np.arange(rows) % len(devices)],
            "timestamp": 1_780_000_000.0 + np.arange(rows, dtype=np.float64) * 0.25,
            "behavior_label": behavior,
            "raw_attack": [raw_map[item] for item in behavior],
        }
    )
    values: dict[str, np.ndarray] = {}
    values["packet_rate"] = rng.lognormal(1.6, 0.45, rows)
    values["byte_rate"] = values["packet_rate"] * rng.lognormal(5.4, 0.35, rows)
    values["burst_score"] = rng.gamma(1.5, 0.5, rows)
    protocol = rng.dirichlet([5.0, 3.0, 0.4], size=rows)
    values["tcp_ratio"], values["udp_ratio"], values["icmp_ratio"] = protocol.T
    values["syn_ratio"] = rng.beta(1.0, 15.0, rows)
    values["rst_ratio"] = rng.beta(1.0, 25.0, rows)
    values["ack_only_ratio"] = rng.beta(2.0, 8.0, rows)
    values["unique_destination_ip_ratio"] = rng.beta(1.5, 8.0, rows)
    values["unique_destination_port_ratio"] = rng.beta(1.5, 9.0, rows)
    values["new_destination_ratio"] = rng.beta(1.5, 10.0, rows)
    values["repeated_destination_score"] = rng.beta(2.0, 5.0, rows)
    values["repeated_port_score"] = rng.beta(2.0, 5.0, rows)
    values["interarrival_mean"] = rng.lognormal(-1.0, 0.5, rows)
    values["interarrival_jitter"] = rng.lognormal(-1.7, 0.6, rows)
    values["interarrival_stability"] = rng.beta(3.0, 3.0, rows)
    values["periodicity_score"] = rng.beta(1.5, 6.0, rows)
    values["outbound_packet_ratio"] = rng.beta(5.0, 3.0, rows)
    values["outbound_byte_ratio"] = rng.beta(4.0, 3.0, rows)
    values["short_flow_ratio"] = rng.beta(3.0, 4.0, rows)
    values["long_flow_ratio"] = rng.beta(1.0, 10.0, rows)
    values["small_packet_ratio"] = rng.beta(4.0, 4.0, rows)
    values["failed_connection_score"] = rng.beta(1.0, 15.0, rows)

    scan = behavior == "scan_like"
    values["packet_rate"][scan] *= 2.0
    values["syn_ratio"][scan] = rng.uniform(0.55, 0.95, scan.sum())
    values["unique_destination_ip_ratio"][scan] = rng.uniform(0.60, 1.0, scan.sum())
    values["unique_destination_port_ratio"][scan] = rng.uniform(0.55, 1.0, scan.sum())
    values["new_destination_ratio"][scan] = rng.uniform(0.65, 1.0, scan.sum())
    values["short_flow_ratio"][scan] = rng.uniform(0.70, 1.0, scan.sum())
    values["failed_connection_score"][scan] = rng.uniform(0.45, 0.95, scan.sum())

    flood = behavior == "flood_like"
    values["packet_rate"][flood] *= rng.uniform(12.0, 30.0, flood.sum())
    values["byte_rate"][flood] *= rng.uniform(8.0, 20.0, flood.sum())
    values["burst_score"][flood] += rng.uniform(4.0, 10.0, flood.sum())
    values["udp_ratio"][flood] = rng.uniform(0.70, 1.0, flood.sum())
    values["tcp_ratio"][flood] *= 0.2

    beacon = behavior == "beacon_like"
    values["repeated_destination_score"][beacon] = rng.uniform(0.80, 1.0, beacon.sum())
    values["repeated_port_score"][beacon] = rng.uniform(0.75, 1.0, beacon.sum())
    values["interarrival_stability"][beacon] = rng.uniform(0.85, 1.0, beacon.sum())
    values["periodicity_score"][beacon] = rng.uniform(0.80, 1.0, beacon.sum())
    values["interarrival_jitter"][beacon] *= 0.08

    exfil = behavior == "exfil_like"
    values["byte_rate"][exfil] *= rng.uniform(8.0, 18.0, exfil.sum())
    values["outbound_packet_ratio"][exfil] = rng.uniform(0.88, 1.0, exfil.sum())
    values["outbound_byte_ratio"][exfil] = rng.uniform(0.92, 1.0, exfil.sum())
    values["long_flow_ratio"][exfil] = rng.uniform(0.65, 1.0, exfil.sum())

    novel = behavior == "unknown_like"
    values["periodicity_score"][novel] = rng.uniform(0.55, 0.78, novel.sum())
    values["new_destination_ratio"][novel] = rng.uniform(0.45, 0.68, novel.sum())
    values["interarrival_mean"][novel] *= 5.0
    values["outbound_byte_ratio"][novel] = rng.uniform(0.72, 0.90, novel.sum())
    values["failed_connection_score"][novel] = rng.uniform(0.30, 0.55, novel.sum())

    for feature in COMMON_STREAM_FEATURES:
        frame[feature] = np.asarray(values[feature], dtype=np.float32)
    ratio_columns = [
        feature
        for feature in COMMON_STREAM_FEATURES
        if feature.endswith("ratio") or feature.endswith("score") or feature.endswith("stability")
    ]
    for column in ratio_columns:
        if column != "burst_score":
            frame[column] = frame[column].clip(0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame

