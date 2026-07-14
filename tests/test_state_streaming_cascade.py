from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bitguard_bnn.cascade import (
    CascadeCalibration,
    route_with_temporal_state,
    tune_exit_threshold,
)
from bitguard_bnn.constants import CANONICAL_LABELS
from bitguard_bnn.config import DEFAULTS
from bitguard_bnn.constants import COMMON_STREAM_FEATURES
from bitguard_bnn.state import TemporalSecurityStateMachine
from bitguard_bnn.streaming import (
    MicroSecurityFeatureProcessor,
    PacketMetadata,
    process_metadata_csv,
)


class StateStreamingCascadeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = copy.deepcopy(DEFAULTS)

    def test_counter_saturates_and_resets(self) -> None:
        machine = TemporalSecurityStateMachine(self.config)
        probability = {"benign": 0.0, "scan_like": 1.0}
        for _ in range(20):
            state, _, _ = machine.update("device", probability)
        self.assertEqual(state.scan, 15)
        self.assertEqual(state.benign, 0)
        machine.reset("device")
        self.assertEqual(machine.get("device").scan, 0)

    def test_wall_clock_decay_accumulates_subintervals(self) -> None:
        machine = TemporalSecurityStateMachine(self.config)
        probability = {"benign": 0.0, "scan_like": 1.0}
        machine.update("device", probability, 0.0)
        machine.update("device", probability, 30.0)
        state, _, _ = machine.update("device", probability, 61.0)
        self.assertEqual(state.scan, 5)

    def test_temporal_device_lru_is_bounded(self) -> None:
        self.config["temporal"]["max_devices"] = 2
        machine = TemporalSecurityStateMachine(self.config)
        for device in ("a", "b", "c"):
            machine.update(device, {"benign": 1.0})
        self.assertEqual(len(machine.states), 2)
        self.assertEqual(machine.evictions, 1)

    def test_streaming_processor_is_payload_free_and_bounded(self) -> None:
        processor = MicroSecurityFeatureProcessor(max_events_per_device=4)
        result = None
        for index in range(8):
            result = processor.update(
                PacketMetadata(
                    device_id="device",
                    timestamp=float(index),
                    length_bytes=80,
                    protocol="tcp",
                    destination_ip=f"10.0.0.{index}",
                    destination_port=20 + index,
                    tcp_flags="SYN",
                    connection_failed=True,
                )
            )
        assert result is not None
        self.assertEqual(set(result), set(COMMON_STREAM_FEATURES))
        self.assertLessEqual(len(processor.devices["device"].events), 4)
        self.assertGreater(result["syn_ratio"], 0.9)

    def test_rst_is_not_counted_as_syn_and_device_lru_is_bounded(self) -> None:
        processor = MicroSecurityFeatureProcessor(max_devices=1)
        result = processor.update(
            PacketMetadata(
                device_id="first",
                timestamp=0.0,
                length_bytes=80,
                protocol="tcp",
                destination_ip="10.0.0.1",
                destination_port=1,
                tcp_flags="RST",
            )
        )
        self.assertEqual(result["syn_ratio"], 0.0)
        self.assertEqual(result["rst_ratio"], 1.0)
        processor.update(
            PacketMetadata(
                device_id="second",
                timestamp=1.0,
                length_bytes=80,
                protocol="udp",
                destination_ip="10.0.0.2",
                destination_port=2,
            )
        )
        self.assertEqual(len(processor.devices), 1)
        self.assertEqual(processor.device_evictions, 1)

    def test_metadata_csv_is_processed_in_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "packets.csv"
            output = root / "features.csv"
            pd.DataFrame(
                {
                    "device_id": ["d", "d", "d"],
                    "timestamp": [0.0, 1.0, 2.0],
                    "length_bytes": [80, 90, 100],
                    "protocol": ["tcp", "tcp", "udp"],
                    "destination_ip": ["a", "b", "b"],
                    "destination_port": [1, 2, 2],
                }
            ).to_csv(source, index=False)
            summary = process_metadata_csv(source, output, chunk_size=2)
            self.assertEqual(summary["rows"], 3)
            result = pd.read_csv(output)
            self.assertEqual(len(result), 3)
            self.assertTrue(set(COMMON_STREAM_FEATURES).issubset(result.columns))

    def test_cascade_threshold_respects_attack_recall(self) -> None:
        benign_probability = np.asarray([0.99, 0.95, 0.91, 0.65, 0.20, 0.10])
        labels = np.asarray(["benign", "benign", "benign", "scan_like", "flood_like", "scan_like"])
        calibration = tune_exit_threshold(benign_probability, labels, 1.0, 51, 0.05)
        self.assertGreaterEqual(calibration.attack_escalation_recall, 1.0)
        self.assertGreater(calibration.benign_early_exit_ratio, 0.0)

    def test_runtime_false_negative_cost_conservatively_escalates(self) -> None:
        metadata = pd.DataFrame(
            {"device_id": ["d"], "source_file": ["capture"], "sequence_index": [0]}
        )
        main = np.zeros((1, len(CANONICAL_LABELS)), dtype=np.float32)
        main[0, CANONICAL_LABELS.index("scan_like")] = 1.0
        no_cost = CascadeCalibration(0.75, 1.0, 0.0, 0.0, 1, 0.0)
        with_cost = CascadeCalibration(0.75, 1.0, 0.0, 0.0, 1, 0.10)
        _, stage_without, _ = route_with_temporal_state(
            metadata, np.asarray([0.9]), main, CANONICAL_LABELS, no_cost, self.config
        )
        _, stage_with, _ = route_with_temporal_state(
            metadata, np.asarray([0.9]), main, CANONICAL_LABELS, with_cost, self.config
        )
        self.assertEqual(stage_without.tolist(), [1])
        self.assertEqual(stage_with.tolist(), [2])


if __name__ == "__main__":
    unittest.main()
