from __future__ import annotations

import math
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
    protocol_counts: Counter[str] = field(default_factory=Counter)
    destination_counts: Counter[str] = field(default_factory=Counter)
    port_counts: Counter[int] = field(default_factory=Counter)
    per_second_counts: Counter[int] = field(default_factory=Counter)
    per_second_count_frequencies: Counter[int] = field(default_factory=Counter)
    maximum_per_second_count: int = 0
    length_sum: float = 0.0
    outbound_length_sum: float = 0.0
    outbound_count: int = 0
    new_destination_count: int = 0
    syn_count: int = 0
    rst_count: int = 0
    ack_only_count: int = 0
    short_flow_count: int = 0
    long_flow_count: int = 0
    small_packet_count: int = 0
    failed_connection_count: int = 0
    interval_sum: float = 0.0
    interval_square_sum: float = 0.0


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
            self._remove_left(state)
        new_destination = event.destination_ip not in state.known_destinations
        if event.destination_ip in state.known_destinations:
            state.known_destinations.move_to_end(event.destination_ip)
        else:
            state.known_destinations[event.destination_ip] = None
            if len(state.known_destinations) > self.max_known_destinations:
                state.known_destinations.popitem(last=False)
        self._append(state, _ObservedEvent(event, new_destination))
        features = self._features(state)
        self.update_latencies_ns.append(time.perf_counter_ns() - started)
        if len(self.update_latencies_ns) > 10_000:
            self.update_latencies_ns = self.update_latencies_ns[-10_000:]
        return features

    def _append(self, state: _DeviceWindow, observed: _ObservedEvent) -> None:
        if len(state.events) >= self.max_events:
            self._remove_left(state)
        event = observed.event
        if state.events:
            interval = event.timestamp - state.events[-1].event.timestamp
            state.interval_sum += interval
            state.interval_square_sum += interval * interval
        state.events.append(observed)
        state.protocol_counts[event.protocol.strip().lower()] += 1
        state.destination_counts[event.destination_ip] += 1
        state.port_counts[int(event.destination_port)] += 1
        second = int(event.timestamp)
        previous_second_count = state.per_second_counts[second]
        if previous_second_count:
            _decrement(state.per_second_count_frequencies, previous_second_count)
        current_second_count = previous_second_count + 1
        state.per_second_counts[second] = current_second_count
        state.per_second_count_frequencies[current_second_count] += 1
        state.maximum_per_second_count = max(
            state.maximum_per_second_count, current_second_count
        )
        state.length_sum += event.length_bytes
        state.outbound_length_sum += event.length_bytes if event.outbound else 0.0
        state.outbound_count += int(event.outbound)
        state.new_destination_count += int(observed.new_destination)
        state.syn_count += int(_has_tcp_flag(event.tcp_flags, "S", "SYN"))
        state.rst_count += int(_has_tcp_flag(event.tcp_flags, "R", "RST"))
        state.ack_only_count += int(_is_ack_only(event.tcp_flags))
        state.short_flow_count += int(event.flow_duration_seconds < 1.0)
        state.long_flow_count += int(event.flow_duration_seconds > 30.0)
        state.small_packet_count += int(event.length_bytes < 128.0)
        state.failed_connection_count += int(event.connection_failed)

    def _remove_left(self, state: _DeviceWindow) -> None:
        observed = state.events.popleft()
        event = observed.event
        if state.events:
            interval = state.events[0].event.timestamp - event.timestamp
            state.interval_sum -= interval
            state.interval_square_sum -= interval * interval
        _decrement(state.protocol_counts, event.protocol.strip().lower())
        _decrement(state.destination_counts, event.destination_ip)
        _decrement(state.port_counts, int(event.destination_port))
        second = int(event.timestamp)
        previous_second_count = state.per_second_counts[second]
        _decrement(state.per_second_count_frequencies, previous_second_count)
        _decrement(state.per_second_counts, second)
        current_second_count = previous_second_count - 1
        if current_second_count:
            state.per_second_count_frequencies[current_second_count] += 1
        if (
            previous_second_count == state.maximum_per_second_count
            and previous_second_count not in state.per_second_count_frequencies
        ):
            state.maximum_per_second_count = current_second_count
        state.length_sum -= event.length_bytes
        state.outbound_length_sum -= event.length_bytes if event.outbound else 0.0
        state.outbound_count -= int(event.outbound)
        state.new_destination_count -= int(observed.new_destination)
        state.syn_count -= int(_has_tcp_flag(event.tcp_flags, "S", "SYN"))
        state.rst_count -= int(_has_tcp_flag(event.tcp_flags, "R", "RST"))
        state.ack_only_count -= int(_is_ack_only(event.tcp_flags))
        state.short_flow_count -= int(event.flow_duration_seconds < 1.0)
        state.long_flow_count -= int(event.flow_duration_seconds > 30.0)
        state.small_packet_count -= int(event.length_bytes < 128.0)
        state.failed_connection_count -= int(event.connection_failed)

    def _features(self, state: _DeviceWindow) -> dict[str, float]:
        event_count = len(state.events)
        count = max(event_count, 1)
        interval_count = max(event_count - 1, 0)
        if event_count >= 2:
            duration = max(
                float(state.events[-1].event.timestamp - state.events[0].event.timestamp),
                1e-3,
            )
            mean_interval = state.interval_sum / interval_count
            interval_variance = max(
                state.interval_square_sum / interval_count - mean_interval * mean_interval,
                0.0,
            )
            jitter = math.sqrt(interval_variance)
            stability = 1.0 / (1.0 + jitter / max(mean_interval, 1e-6))
        else:
            duration = max(self.window_seconds, 1e-3)
            mean_interval = 0.0
            jitter = 0.0
            stability = 0.0
        average_bin = count / max(len(state.per_second_counts), 1)
        maximum_bin = state.maximum_per_second_count
        feature_values = {
            "packet_rate": count / duration,
            "byte_rate": state.length_sum / duration,
            "burst_score": maximum_bin / max(average_bin, 1e-6),
            "tcp_ratio": state.protocol_counts["tcp"] / count,
            "udp_ratio": state.protocol_counts["udp"] / count,
            "icmp_ratio": state.protocol_counts["icmp"] / count,
            "syn_ratio": state.syn_count / count,
            "rst_ratio": state.rst_count / count,
            "ack_only_ratio": state.ack_only_count / count,
            "unique_destination_ip_ratio": len(state.destination_counts) / count,
            "unique_destination_port_ratio": len(state.port_counts) / count,
            "new_destination_ratio": state.new_destination_count / count,
            "repeated_destination_score": 1.0 - len(state.destination_counts) / count,
            "repeated_port_score": 1.0 - len(state.port_counts) / count,
            "interarrival_mean": mean_interval,
            "interarrival_jitter": jitter,
            "interarrival_stability": stability,
            "periodicity_score": stability if interval_count >= 3 else 0.0,
            "outbound_packet_ratio": state.outbound_count / count,
            "outbound_byte_ratio": (
                state.outbound_length_sum / max(state.length_sum, 1.0) if event_count else 0.0
            ),
            "short_flow_ratio": state.short_flow_count / count,
            "long_flow_ratio": state.long_flow_count / count,
            "small_packet_ratio": state.small_packet_count / count,
            "failed_connection_score": state.failed_connection_count / count,
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


def _decrement(counter: Counter, key: object) -> None:
    counter[key] -= 1
    if counter[key] <= 0:
        del counter[key]


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
