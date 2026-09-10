from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bitguard_bnn.models import BNNClassifier
from bitguard_bnn.rbc_edge import (
    PackedRBCRuntime,
    calibrate_detection_threshold,
    export_rbc_edge,
)


class _DetectionFirstBNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = BNNClassifier(5, [4, 3], 1)
        self.type_head = nn.Linear(3, 1)

    def hidden(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.detector.blocks:
            values = block(values)
        return values


class RBCEdgeTest(unittest.TestCase):
    def _model(self) -> _DetectionFirstBNN:
        torch.manual_seed(19)
        model = _DetectionFirstBNN().eval()
        with torch.no_grad():
            # Exercise negative and exactly-zero BN scales as well as reachable boundaries.
            model.detector.blocks[0][1].weight.copy_(torch.tensor([1.0, -1.0, 0.0, -0.5]))
            model.detector.blocks[0][1].bias.copy_(torch.tensor([0.0, 0.0, -0.2, 1.0]))
            model.detector.blocks[0][1].running_mean.copy_(
                torch.tensor([1.0, 1.0, 0.0, -1.0])
            )
            model.detector.blocks[0][1].running_var.fill_(1.0)
        return model

    def test_packed_hidden_matches_shared_blocks_for_all_first_layer_inputs(self) -> None:
        model = self._model()
        values = np.asarray(
            [[1.0 if bits & (1 << column) else -1.0 for column in range(5)] for bits in range(32)],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rbc.npz"
            metadata = export_rbc_edge(model, path)
            runtime = PackedRBCRuntime.load(path)
            with torch.inference_mode():
                expected = model.hidden(torch.from_numpy(values)).numpy().astype(np.int8)
            np.testing.assert_array_equal(runtime.hidden(values), expected)
            self.assertEqual(metadata["static_data_bytes"], path.stat().st_size)
            self.assertEqual(runtime.static_data_bytes, path.stat().st_size)

    def test_integer_scores_use_exported_int8_weights_and_int32_biases(self) -> None:
        model = self._model()
        values = np.asarray([[1, -1, 1, -1, 1], [-1, 1, -1, 1, -1]], dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rbc.npz"
            export_rbc_edge(model, path, detection_threshold=0, type_labels=["scan", "flood"])
            runtime = PackedRBCRuntime.load(path)
            result = runtime.predict(values)
            hidden = runtime.hidden(values).astype(np.int32)
            with np.load(path, allow_pickle=False) as arrays:
                expected_detection = (
                    hidden @ arrays["detection_weight_int8"].astype(np.int32).T
                    + arrays["detection_bias_int32"]
                )
                expected_type = (
                    hidden @ arrays["type_weight_int8"].astype(np.int32).T
                    + arrays["type_bias_int32"]
                )
                self.assertEqual(arrays["detection_weight_int8"].dtype, np.int8)
                self.assertEqual(arrays["detection_bias_int32"].dtype, np.int32)
            np.testing.assert_array_equal(result.detection_scores, expected_detection)
            np.testing.assert_array_equal(result.type_scores, expected_type)
            np.testing.assert_array_equal(result.is_attack, expected_detection[:, 0] > 0)

    def test_calibration_uses_strict_integer_score_boundary(self) -> None:
        scores = np.asarray([9, 9, 7, 3, 8, 1], dtype=np.int32)
        labels = np.asarray([False, False, False, False, True, True])
        threshold = calibrate_detection_threshold(
            scores, labels, maximum_false_positive_rate=0.25
        )
        self.assertEqual(threshold, 9)
        self.assertEqual(int(np.count_nonzero(scores[~labels] > threshold)), 0)


if __name__ == "__main__":
    unittest.main()
