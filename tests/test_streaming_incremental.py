from __future__ import annotations

import math
import re
import unittest
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np

from bitguard_bnn.constants import COMMON_STREAM_FEATURES
from bitguard_bnn.streaming import MicroSecurityFeatureProcessor, PacketMetadata


@dataclass(frozen=True)
class _ReferenceObserved:
    event: PacketMetadata
    new_destination: bool


class _SlowReference:
    """Independent full-window implementation used to lock feature semantics."""

    def __init__(self, window_seconds: float, max_events: int, max_known: int) -> None:
        self.window_seconds = float(window_seconds)
        self.events: deque[_ReferenceObserved] = deque(maxlen=max_events)
        self.known: OrderedDict[str, None] = OrderedDict()
        self.max_known = max_known

    def update(self, event: PacketMetadata) -> dict[str, float]:
        cutoff = event.timestamp - self.window_seconds
        while self.events and self.events[0].event.timestamp < cutoff:
            self.events.popleft()

        new_destination = event.destination_ip not in self.known
        if event.destination_ip in self.known:
            self.known.move_to_end(event.destination_ip)
        else:
            self.known[event.destination_ip] = None
            if len(self.known) > self.max_known:
                self.known.popitem(last=False)
        self.events.append(_ReferenceObserved(event, new_destination))
        return self._features()

    def _features(self) -> dict[str, float]:
        rows = list(self.events)
        events = [row.event for row in rows]
        count = max(len(events), 1)
        if len(events) >= 2:
            intervals = [
                current.timestamp - previous.timestamp
                for previous, current in zip(events, events[1:])
            ]
            duration = max(events[-1].timestamp - events[0].timestamp, 1e-3)
        else:
            intervals = []
            duration = max(self.window_seconds, 1e-3)

        length_sum = float(sum(event.length_bytes for event in events))
        destinations = Counter(event.destination_ip for event in events)
        ports = Counter(int(event.destination_port) for event in events)
        bins = Counter(int(event.timestamp) for event in events)
        average_bin = count / max(len(bins), 1)
        maximum_bin = max(bins.values(), default=0)
        mean_interval = sum(intervals) / len(intervals) if intervals else 0.0
        jitter = (
            math.sqrt(sum((value - mean_interval) ** 2 for value in intervals) / len(intervals))
            if intervals
            else 0.0
        )
        stability = (
            1.0 / (1.0 + jitter / max(mean_interval, 1e-6)) if intervals else 0.0
        )
        outbound = [event for event in events if event.outbound]

        values = {
            "packet_rate": count / duration,
            "byte_rate": length_sum / duration,
            "burst_score": maximum_bin / max(average_bin, 1e-6),
            "tcp_ratio": sum(event.protocol.strip().lower() == "tcp" for event in events) / count,
            "udp_ratio": sum(event.protocol.strip().lower() == "udp" for event in events) / count,
            "icmp_ratio": sum(event.protocol.strip().lower() == "icmp" for event in events) / count,
            "syn_ratio": sum(_has_flag(event.tcp_flags, "S", "SYN") for event in events) / count,
            "rst_ratio": sum(_has_flag(event.tcp_flags, "R", "RST") for event in events) / count,
            "ack_only_ratio": sum(_ack_only(event.tcp_flags) for event in events) / count,
            "unique_destination_ip_ratio": len(destinations) / count,
            "unique_destination_port_ratio": len(ports) / count,
            "new_destination_ratio": sum(row.new_destination for row in rows) / count,
            "repeated_destination_score": 1.0 - len(destinations) / count,
            "repeated_port_score": 1.0 - len(ports) / count,
            "interarrival_mean": mean_interval,
            "interarrival_jitter": jitter,
            "interarrival_stability": stability,
            "periodicity_score": stability if len(intervals) >= 3 else 0.0,
            "outbound_packet_ratio": len(outbound) / count,
            "outbound_byte_ratio": (
                sum(event.length_bytes for event in outbound) / max(length_sum, 1.0)
                if events
                else 0.0
            ),
            "short_flow_ratio": sum(event.flow_duration_seconds < 1.0 for event in events) / count,
            "long_flow_ratio": sum(event.flow_duration_seconds > 30.0 for event in events) / count,
            "small_packet_ratio": sum(event.length_bytes < 128.0 for event in events) / count,
            "failed_connection_score": sum(event.connection_failed for event in events) / count,
        }
        return {name: float(values[name]) for name in COMMON_STREAM_FEATURES}


def _flag_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^A-Z0-9]+", value.upper()) if token}


def _has_flag(value: str, short: str, long: str) -> bool:
    tokens = _flag_tokens(value)
    if long in tokens or short in tokens:
        return True
    return any(len(token) <= 2 and short in token for token in tokens)


def _ack_only(value: str) -> bool:
    return _flag_tokens(value) in ({"A"}, {"ACK"})


def _event(index: int, timestamp: float) -> PacketMetadata:
    protocols = ("tcp", "udp", "icmp", " TCP ")
    flags = ("SYN", "RST", "ACK", "S A")
    return PacketMetadata(
        device_id="device",
        timestamp=timestamp,
        length_bytes=60 + index * 37,
        protocol=protocols[index % len(protocols)],
        destination_ip=f"10.0.0.{index % 3}",
        destination_port=20 + index % 4,
        tcp_flags=flags[index % len(flags)],
        outbound=index % 3 != 0,
        flow_duration_seconds=(0.2, 4.0, 45.0)[index % 3],
        connection_failed=index % 2 == 0,
    )


class IncrementalStreamingTest(unittest.TestCase):
    def assert_features_equal(
        self, actual: dict[str, float], expected: dict[str, float]
    ) -> None:
        self.assertEqual(set(actual), set(COMMON_STREAM_FEATURES))
        self.assertEqual(set(actual), set(expected))
        for name in COMMON_STREAM_FEATURES:
            self.assertAlmostEqual(actual[name], expected[name], places=8, msg=name)

    def test_incremental_updates_match_slow_reference_through_expiry_and_capacity(self) -> None:
        processor = MicroSecurityFeatureProcessor(
            window_seconds=5.0,
            max_events_per_device=4,
            max_known_destinations=3,
        )
        reference = _SlowReference(5.0, 4, 3)

        # 20 expires the initial window, 19 is deliberately out of order, and
        # the final events exercise the fixed-capacity left eviction path.
        for index, timestamp in enumerate((10.0, 11.0, 12.0, 20.0, 19.0, 21.0, 22.0, 23.0)):
            event = _event(index, timestamp)
            self.assert_features_equal(processor.update(event), reference.update(event))

        self.assertEqual(len(processor.devices["device"].events), 4)

    def test_update_does_not_rebuild_numpy_arrays_from_the_active_window(self) -> None:
        processor = MicroSecurityFeatureProcessor(max_events_per_device=128)
        original_asarray = np.asarray
        with patch("bitguard_bnn.streaming.np.asarray", wraps=original_asarray) as asarray:
            for index in range(16):
                processor.update(_event(index, float(index)))
        self.assertEqual(asarray.call_count, 0)

    def test_update_does_not_scan_all_per_second_bins_for_the_maximum(self) -> None:
        processor = MicroSecurityFeatureProcessor(max_events_per_device=128)
        with patch.object(
            Counter,
            "values",
            side_effect=AssertionError("per-second bin scan is not constant time"),
        ):
            processor.update(_event(0, 0.0))


if __name__ == "__main__":
    unittest.main()
