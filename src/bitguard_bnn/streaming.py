from __future__ import annotations

import re
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

import numpy as np
import pandas as pd

from .constants import COMMON_STREAM_FEATURES


@dataclass(frozen=True)
class PacketMetadata:
    device_id: str
    timestamp: float
    length_bytes: int
    protocol: str
    destination_ip: str
    destination_port: int
    tcp_flags: str = ""
    outbound: bool = True
    flow_duration_seconds: float = 0.0
    connection_failed: bool = False


@dataclass
class _ObservedEvent:
    event: PacketMetadata
    new_destination: bool


@dataclass
class _DeviceWindow:
    events: Deque[_ObservedEvent]
    known_destinations: OrderedDict[str, None] = field(default_factory=OrderedDict)


class MicroSecurityFeatureProcessor:
    """Bounded, payload-free reference feature processor.

    Python is used for correctness experiments. A target implementation should
    replace strings and dictionaries with fixed-width hashes/counters.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        max_events_per_device: int = 2048,
        max_known_destinations: int = 512,
        max_devices: int = 4096,
    ) -> None:
        self.window_seconds = float(window_seconds)
        self.max_events = int(max_events_per_device)
        self.max_known_destinations = int(max_known_destinations)
        self.max_devices = int(max_devices)
        if self.max_events <= 0 or self.max_known_destinations <= 0 or self.max_devices <= 0:
            raise ValueError("streaming capacity limits must be positive")
        self.devices: OrderedDict[str, _DeviceWindow] = OrderedDict()
        self.device_evictions = 0
        self.update_latencies_ns: list[int] = []

    def reset(self, device_id: str | None = None) -> None:
        if device_id is None:
            self.devices.clear()
        else:
            self.devices.pop(device_id, None)

    def update(self, event: PacketMetadata) -> dict[str, float]:
        started = time.perf_counter_ns()
        if event.device_id in self.devices:
            self.devices.move_to_end(event.device_id)
            state = self.devices[event.device_id]
        else:
            if len(self.devices) >= self.max_devices:
                self.devices.popitem(last=False)
                self.device_evictions += 1
            state = _DeviceWindow(deque(maxlen=self.max_events))
            self.devices[event.device_id] = state
        cutoff = event.timestamp - self.window_seconds
        while state.events and state.events[0].event.timestamp < cutoff:
            state.events.popleft()
        new_destination = event.destination_ip not in state.known_destinations
        if event.destination_ip in state.known_destinations:
            state.known_destinations.move_to_end(event.destination_ip)
        else:
            state.known_destinations[event.destination_ip] = None
            if len(state.known_destinations) > self.max_known_destinations:
                state.known_destinations.popitem(last=False)
        state.events.append(_ObservedEvent(event, new_destination))
        features = self._features(state.events)
        self.update_latencies_ns.append(time.perf_counter_ns() - started)
        if len(self.update_latencies_ns) > 10_000:
            self.update_latencies_ns = self.update_latencies_ns[-10_000:]
        return features

    def _features(self, observed: Deque[_ObservedEvent]) -> dict[str, float]:
        rows = list(observed)
        events = [item.event for item in rows]
        count = max(len(events), 1)
        if len(events) >= 2:
            timestamps = np.asarray([event.timestamp for event in events], dtype=np.float64)
            interval = np.diff(timestamps)
            duration = max(float(timestamps[-1] - timestamps[0]), 1e-3)
        else:
            interval = np.asarray([], dtype=np.float64)
            duration = max(self.window_seconds, 1e-3)
        lengths = np.asarray([event.length_bytes for event in events], dtype=np.float64)
        protocols = [event.protocol.strip().lower() for event in events]
        flags = [event.tcp_flags.upper() for event in events]
        destinations = [event.destination_ip for event in events]
        ports = [int(event.destination_port) for event in events]
        destination_counts = Counter(destinations)
        port_counts = Counter(ports)
        outbound_mask = np.asarray([event.outbound for event in events], dtype=bool)
        flow_duration = np.asarray([event.flow_duration_seconds for event in events], dtype=np.float64)
        per_second = Counter(int(event.timestamp) for event in events)
        average_bin = count / max(len(per_second), 1)
        maximum_bin = max(per_second.values(), default=0)
        mean_interval = float(interval.mean()) if len(interval) else 0.0
        jitter = float(interval.std()) if len(interval) else 0.0
        stability = 1.0 / (1.0 + jitter / max(mean_interval, 1e-6)) if len(interval) else 0.0
        feature_values = {
            "packet_rate": count / duration,
            "byte_rate": float(lengths.sum()) / duration,
            "burst_score": maximum_bin / max(average_bin, 1e-6),
            "tcp_ratio": protocols.count("tcp") / count,
            "udp_ratio": protocols.count("udp") / count,
            "icmp_ratio": protocols.count("icmp") / count,
            "syn_ratio": sum(_has_tcp_flag(value, "S", "SYN") for value in flags) / count,
            "rst_ratio": sum(_has_tcp_flag(value, "R", "RST") for value in flags) / count,
            "ack_only_ratio": sum(_is_ack_only(value) for value in flags) / count,
            "unique_destination_ip_ratio": len(destination_counts) / count,
            "unique_destination_port_ratio": len(port_counts) / count,
            "new_destination_ratio": sum(item.new_destination for item in rows) / count,
            "repeated_destination_score": 1.0 - len(destination_counts) / count,
            "repeated_port_score": 1.0 - len(port_counts) / count,
            "interarrival_mean": mean_interval,
            "interarrival_jitter": jitter,
            "interarrival_stability": stability,
            "periodicity_score": stability if len(interval) >= 3 else 0.0,
            "outbound_packet_ratio": float(outbound_mask.mean()) if len(outbound_mask) else 0.0,
            "outbound_byte_ratio": (
                float(lengths[outbound_mask].sum() / max(lengths.sum(), 1.0)) if len(lengths) else 0.0
            ),
            "short_flow_ratio": float(np.mean(flow_duration < 1.0)) if len(flow_duration) else 0.0,
            "long_flow_ratio": float(np.mean(flow_duration > 30.0)) if len(flow_duration) else 0.0,
            "small_packet_ratio": float(np.mean(lengths < 128.0)) if len(lengths) else 0.0,
            "failed_connection_score": sum(event.connection_failed for event in events) / count,
        }
        return {name: float(feature_values[name]) for name in COMMON_STREAM_FEATURES}

    def latency_summary(self) -> dict[str, float | int]:
        if not self.update_latencies_ns:
            return {
                "samples": 0,
                "p50_microseconds": 0.0,
                "p95_microseconds": 0.0,
                "device_evictions": self.device_evictions,
                "max_devices": self.max_devices,
            }
        values = np.asarray(self.update_latencies_ns, dtype=np.float64) / 1_000.0
        return {
            "samples": len(values),
            "p50_microseconds": float(np.percentile(values, 50)),
            "p95_microseconds": float(np.percentile(values, 95)),
            "device_evictions": self.device_evictions,
            "max_devices": self.max_devices,
        }


def process_metadata_csv(
    input_path: Path,
    output_path: Path,
    *,
    window_seconds: float = 60.0,
    max_events_per_device: int = 2048,
    max_devices: int = 4096,
    chunk_size: int = 100_000,
) -> dict[str, float | int | str]:
    """Convert an ordered packet-metadata CSV to the shared feature schema."""

    required = {
        "device_id",
        "timestamp",
        "length_bytes",
        "protocol",
        "destination_ip",
        "destination_port",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processor = MicroSecurityFeatureProcessor(
        window_seconds, max_events_per_device, max_devices=max_devices
    )
    row_count = 0
    previous_timestamp = -np.inf
    wrote_header = False
    for frame in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"metadata CSV is missing columns: {sorted(missing)}")
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise")
        if not frame["timestamp"].is_monotonic_increasing or (
            len(frame) and float(frame["timestamp"].iloc[0]) < previous_timestamp
        ):
            raise ValueError("metadata CSV must be globally ordered by timestamp before streaming")
        if len(frame):
            previous_timestamp = float(frame["timestamp"].iloc[-1])
        preserved = [
            column
            for column in ("behavior_label", "raw_attack", "flow_id", "capture_id")
            if column in frame.columns
        ]
        rows: list[dict[str, object]] = []
        for record in frame.to_dict(orient="records"):
            event = PacketMetadata(
                device_id=str(record["device_id"]),
                timestamp=float(record["timestamp"]),
                length_bytes=int(record["length_bytes"]),
                protocol=str(record["protocol"]),
                destination_ip=str(record["destination_ip"]),
                destination_port=int(record["destination_port"]),
                tcp_flags=str(record.get("tcp_flags", "")),
                outbound=_as_bool(record.get("outbound", True), default=True),
                flow_duration_seconds=_as_float(record.get("flow_duration_seconds", 0.0)),
                connection_failed=_as_bool(record.get("connection_failed", False), default=False),
            )
            rows.append(
                {
                    "device_id": event.device_id,
                    "timestamp": event.timestamp,
                    **{column: record[column] for column in preserved},
                    **processor.update(event),
                }
            )
        output_frame = pd.DataFrame(rows)
        output_frame.to_csv(
            output_path,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True
        row_count += len(output_frame)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": row_count,
        **processor.latency_summary(),
    }


def _as_bool(value: object, default: bool) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "outbound", "out"}
    return bool(value)


def _as_float(value: object, default: float = 0.0) -> float:
    return default if pd.isna(value) else float(value)


def _flag_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^A-Z0-9]+", value.upper()) if token}


def _has_tcp_flag(value: str, short: str, long: str) -> bool:
    tokens = _flag_tokens(value)
    if long in tokens or short in tokens:
        return True
    return any(len(token) <= 2 and short in token for token in tokens)


def _is_ack_only(value: str) -> bool:
    tokens = _flag_tokens(value)
    return tokens in ({"A"}, {"ACK"})
