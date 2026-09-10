from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray



FORMAT_VERSION = 1


@dataclass(frozen=True)
class RBCInference:
    """Integer outputs produced by the packed CPU runtime."""

    detection_scores: NDArray[np.int32]
    type_scores: NDArray[np.int32]
    is_attack: NDArray[np.bool_] | None
    attack_types: NDArray[np.int64]

    @property
    def attack_scores(self) -> NDArray[np.int32]:
        if self.detection_scores.shape[1] == 1:
            return self.detection_scores[:, 0]
        return (self.detection_scores[:, 1] - self.detection_scores[:, 0]).astype(np.int32)

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Keep the compact ``scores, type_indices = runtime.predict(...)`` API."""

        yield self.attack_scores
        yield self.attack_types


def _shared_blocks(model: Any) -> Sequence[Any]:
    blocks = getattr(model, "shared_blocks", None)
    if blocks is None:
        blocks = getattr(model, "blocks", None)
    if blocks is None:
        detector = getattr(model, "detector", None)
        blocks = getattr(detector, "blocks", None)
    if blocks is None:
        raise TypeError("RBC model must expose detector.blocks, shared_blocks, or blocks")
    return blocks


def _head(model: Any, names: Iterable[str]) -> Any:
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value
    raise TypeError(f"RBC model does not expose any of: {', '.join(names)}")


def _linear_from_head(head: Any) -> Any:
    if hasattr(head, "weight") and hasattr(head, "bias"):
        return head
    modules = list(head.modules()) if hasattr(head, "modules") else []
    linear = [module for module in modules if hasattr(module, "weight") and hasattr(module, "bias")]
    if len(linear) != 1:
        raise TypeError("an RBC output head must contain exactly one linear layer")
    return linear[0]


def _packed_rows(binary_weight: NDArray[np.bool_]) -> NDArray[np.uint8]:
    return np.packbits(binary_weight, axis=1, bitorder="little").astype(np.uint8, copy=False)


def _enumerated_bn_rule(
    linear: Any, batchnorm: Any, *, input_width: int | None = None
) -> tuple[NDArray[np.int16], NDArray[np.int8]]:
    """Return exact popcount cutoffs after enumerating every reachable binary dot product.

    The direction is 1 for ``matches >= cutoff``, -1 for ``matches <= cutoff``, and
    0 for a constant output whose sign is stored in ``cutoff``. Enumeration avoids
    rounding a real-valued folded threshold onto the wrong side of a reachable dot.
    """

    import torch
    from torch.nn import functional as functional
    from .export import fold_linear_batchnorm, folded_sign

    folded = fold_linear_batchnorm(linear, batchnorm)
    width = int(linear.in_features if input_width is None else input_width)
    dots = (2 * np.arange(width + 1, dtype=np.int32) - width).astype(np.float32)
    folded_outputs = folded_sign(
        np.broadcast_to(dots[:, None], (width + 1, int(linear.out_features))), folded
    ).astype(np.int8)
    dot_matrix = np.broadcast_to(
        dots[:, None], (width + 1, int(linear.out_features))
    ).copy()
    if linear.bias is not None:
        dot_matrix += linear.bias.detach().cpu().numpy().astype(np.float32)[None, :]
    with torch.inference_mode():
        normalized = functional.batch_norm(
            torch.from_numpy(dot_matrix),
            batchnorm.running_mean.detach().cpu(),
            batchnorm.running_var.detach().cpu(),
            batchnorm.weight.detach().cpu(),
            batchnorm.bias.detach().cpu(),
            training=False,
            momentum=0.0,
            eps=float(batchnorm.eps),
        )
        outputs = torch.where(normalized >= 0, 1, -1).numpy().astype(np.int8)
    # The existing fold remains the normal fast path. Direct enumeration wins at
    # the rare float32 boundary where division-based thresholds round differently.
    if np.array_equal(folded_outputs, outputs):
        outputs = folded_outputs
    cutoffs = np.empty(int(linear.out_features), dtype=np.int16)
    directions = np.empty(int(linear.out_features), dtype=np.int8)
    for row in range(int(linear.out_features)):
        signs = outputs[:, row]
        positive = np.flatnonzero(signs > 0)
        if positive.size == 0 or positive.size == width + 1:
            directions[row] = 0
            cutoffs[row] = 1 if positive.size else -1
        elif np.all(signs[:-1] <= signs[1:]):
            directions[row] = 1
            cutoffs[row] = int(positive[0])
        elif np.all(signs[:-1] >= signs[1:]):
            directions[row] = -1
            cutoffs[row] = int(positive[-1])
        else:  # BatchNorm followed by sign must be monotone over a scalar dot product.
            raise ValueError("folded BatchNorm produced a non-monotone reachable sign rule")
    return cutoffs, directions


def _quantize_head(linear: Any) -> tuple[NDArray[np.int8], NDArray[np.int32], np.float32]:
    weight = linear.weight.detach().cpu().numpy().astype(np.float32)
    bias = linear.bias.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("output parameters must be finite")
    maximum = float(np.max(np.abs(weight), initial=0.0))
    headroom = np.iinfo(np.int32).max - weight.shape[1] * 127 - 1
    if headroom <= 0:
        raise ValueError("output accumulator exceeds int32 capacity")
    scale = np.nextafter(np.float32(max(maximum / 127.0,
        float(np.max(np.abs(bias), initial=0.0)) / headroom,
        np.finfo(np.float32).tiny)), np.float32(np.inf))
    quantized_weight = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
    quantized_bias = np.rint(bias.astype(np.float64) / float(scale)).astype(np.int32)
    return quantized_weight, quantized_bias, scale


def _save_with_exact_size(path: Path, arrays: dict[str, NDArray[Any]]) -> int:
    """Store the archive size in the archive itself and converge to the exact value."""

    arrays["static_data_bytes"] = np.asarray([0], dtype=np.int64)
    last_size = -1
    for _ in range(16):
        np.savez_compressed(path, **arrays)
        size = path.stat().st_size
        if int(arrays["static_data_bytes"][0]) == size:
            return size
        if size == last_size:
            arrays["static_data_bytes"][0] = size
        else:
            arrays["static_data_bytes"][0] = size
            last_size = size
    np.savez_compressed(path, **arrays)
    size = path.stat().st_size
    if int(arrays["static_data_bytes"][0]) != size:
        raise RuntimeError("could not encode the exact NPZ file size in static_data_bytes")
    return size


def export_rbc_edge(
    model: Any,
    path: str | Path,
    *,
    detection_threshold: int | None = None,
    type_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export a DetectionFirstBNN to a static packed CPU artifact.

    ``detection_threshold`` is deliberately an integer score threshold. Calibration
    therefore occurs against this quantized runtime, after output quantization.
    """

    from .models import BinaryLinear

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[Any]] = {}
    blocks = _shared_blocks(model)
    if not blocks:
        raise ValueError("RBC edge export requires at least one shared binary block")
    detector_model = getattr(model, "detector", model)
    gate = getattr(detector_model, "feature_gate", None)
    original_input_width = int(blocks[0][0].in_features)
    if gate is None:
        active_inputs = np.arange(original_input_width, dtype=np.int64)
    else:
        hard_mask = gate.hard_mask().detach().cpu().numpy().astype(bool, copy=False)
        groups = gate.input_groups.detach().cpu().numpy().astype(np.int64, copy=False)
        active_groups = np.flatnonzero(hard_mask)
        active_inputs = np.flatnonzero(np.isin(groups, active_groups)).astype(np.int64)
        if active_inputs.size == 0:
            raise ValueError("RBC feature gate selected no encoded inputs")
    arrays["runtime_input_bits"] = np.asarray([original_input_width], dtype=np.int32)
    arrays["active_input_indices"] = active_inputs

    for index, block in enumerate(blocks):
        linear, batchnorm = block[0], block[1]
        if not isinstance(linear, BinaryLinear):
            raise TypeError("RBC packed runtime requires BinaryLinear shared blocks")
        signed = linear.weight.detach().cpu().numpy() >= 0
        if index == 0:
            signed = signed[:, active_inputs]
        arrays[f"block_{index}_weight_bits"] = _packed_rows(signed)
        cutoff, direction = _enumerated_bn_rule(
            linear, batchnorm, input_width=int(signed.shape[1])
        )
        arrays[f"block_{index}_cutoff"] = cutoff
        arrays[f"block_{index}_direction"] = direction
        arrays[f"block_{index}_input_bits"] = np.asarray([signed.shape[1]], dtype=np.int32)

    detector = getattr(detector_model, "output", None)
    if detector is None:
        detector = _linear_from_head(
            _head(model, ("detection_head", "detector_head", "detector"))
        )
    type_head = _linear_from_head(_head(model, ("type_head", "attack_type_head")))
    detection_weight, detection_bias, detection_scale = _quantize_head(detector)
    type_weight, type_bias, type_scale = _quantize_head(type_head)
    arrays.update(
        {
            "detection_weight_int8": detection_weight,
            "detection_bias_int32": detection_bias,
            "detection_scale_fp32": np.asarray([detection_scale], dtype=np.float32),
            "type_weight_int8": type_weight,
            "type_bias_int32": type_bias,
            "type_scale_fp32": np.asarray([type_scale], dtype=np.float32),
            "detection_threshold_int32": np.asarray(
                [] if detection_threshold is None else [detection_threshold], dtype=np.int32
            ),
        }
    )
    metadata = {
        "format_version": FORMAT_VERSION,
        "block_count": len(blocks),
        "bit_order": "little",
        "score_arithmetic": "int8 weights, int32 bias and accumulation",
        "detection_rule": "attack_score > detection_threshold_int32",
        "calibration_stage": "after_output_quantization",
        "type_labels": list(type_labels or ()),
        "attack_classes": int(getattr(model, "attack_classes", 2 if type_weight.shape[0] == 1 else type_weight.shape[0])),
    }
    arrays["metadata_json"] = np.asarray(
        [json.dumps(metadata, ensure_ascii=False, sort_keys=True)], dtype=np.str_
    )
    size = _save_with_exact_size(destination, arrays)
    metadata["static_data_bytes"] = size
    return metadata


class PackedRBCRuntime:
    """Dependency-light XNOR/popcount runtime for a packed RBC artifact."""

    def __init__(self, arrays: dict[str, NDArray[Any]], path: Path) -> None:
        metadata = json.loads(str(arrays["metadata_json"][0]))
        if int(metadata["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"unsupported RBC edge format: {metadata['format_version']}")
        expected_size = int(arrays["static_data_bytes"][0])
        actual_size = path.stat().st_size
        if expected_size != actual_size:
            raise ValueError(
                f"RBC edge artifact size mismatch: metadata={expected_size}, actual={actual_size}"
            )
        self.path = path
        self.metadata = metadata | {"static_data_bytes": actual_size}
        self._arrays = arrays
        self._block_count = int(metadata["block_count"])

    @classmethod
    def load(cls, path: str | Path) -> PackedRBCRuntime:
        artifact = Path(path)
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        return cls(arrays, artifact)

    @property
    def static_data_bytes(self) -> int:
        return int(self.metadata["static_data_bytes"])

    @staticmethod
    def _binary(values: NDArray[Any], width: int) -> NDArray[np.bool_]:
        matrix = np.asarray(values)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != width:
            raise ValueError(f"expected a [batch, {width}] binary input")
        valid = np.logical_or(np.logical_or(matrix == -1, matrix == 0), matrix == 1)
        if not bool(np.all(valid)):
            raise ValueError("RBC edge inputs must contain only -1, 0, or 1")
        return matrix > 0

    def hidden(self, values: NDArray[Any]) -> NDArray[np.int8]:
        runtime_width = int(self._arrays["runtime_input_bits"][0])
        activations = self._binary(values, runtime_width)
        activations = activations[:, self._arrays["active_input_indices"]]
        for index in range(self._block_count):
            width = int(self._arrays[f"block_{index}_input_bits"][0])
            if activations.shape[1] != width:
                raise ValueError("RBC artifact has inconsistent shared block dimensions")
            packed = np.packbits(activations, axis=1, bitorder="little")
            weights = self._arrays[f"block_{index}_weight_bits"]
            cutoffs = self._arrays[f"block_{index}_cutoff"]
            directions = self._arrays[f"block_{index}_direction"]
            next_values = np.empty((len(packed), len(weights)), dtype=np.bool_)
            mask = (1 << width) - 1
            weight_rows = [int.from_bytes(row.tobytes(), "little") for row in weights]
            for sample_index, sample in enumerate(packed):
                sample_bits = int.from_bytes(sample.tobytes(), "little")
                for row_index, weight_bits in enumerate(weight_rows):
                    matches = width - ((sample_bits ^ weight_bits) & mask).bit_count()
                    direction = int(directions[row_index])
                    cutoff = int(cutoffs[row_index])
                    next_values[sample_index, row_index] = (
                        cutoff > 0 if direction == 0 else
                        matches >= cutoff if direction > 0 else matches <= cutoff
                    )
            activations = next_values
        return np.where(activations, 1, -1).astype(np.int8)

    def predict(self, values: NDArray[Any]) -> RBCInference:
        hidden = self.hidden(values).astype(np.int32)
        detection_scores = (
            hidden @ self._arrays["detection_weight_int8"].astype(np.int32).T
            + self._arrays["detection_bias_int32"]
        ).astype(np.int32)
        type_scores = (
            hidden @ self._arrays["type_weight_int8"].astype(np.int32).T
            + self._arrays["type_bias_int32"]
        ).astype(np.int32)
        attack_score = (
            detection_scores[:, 0]
            if detection_scores.shape[1] == 1
            else detection_scores[:, 1] - detection_scores[:, 0]
        )
        threshold = self._arrays["detection_threshold_int32"]
        is_attack = None if threshold.size == 0 else attack_score > int(threshold[0])
        attack_types = (
            (type_scores[:, 0] > 0).astype(np.int64)
            if type_scores.shape[1] == 1 and self.metadata.get("attack_classes", 2) == 2
            else np.argmax(type_scores, axis=1).astype(np.int64)
        )
        return RBCInference(
            detection_scores=detection_scores,
            type_scores=type_scores,
            is_attack=is_attack,
            attack_types=attack_types,
        )


def export_rbc(model: Any, path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Public compact alias used by the training/export pipeline."""

    return export_rbc_edge(model, path, **kwargs)


def load_rbc(path: str | Path) -> PackedRBCRuntime:
    """Load a packed RBC artifact."""

    return PackedRBCRuntime.load(path)


def calibrate_detection_threshold(
    integer_attack_scores: NDArray[np.integer[Any]],
    is_attack: NDArray[np.bool_],
    *,
    maximum_false_positive_rate: float,
) -> int:
    """Choose a strict integer threshold using a calibration set only."""

    if not 0.0 <= maximum_false_positive_rate <= 1.0:
        raise ValueError("maximum_false_positive_rate must be between 0 and 1")
    scores = np.asarray(integer_attack_scores, dtype=np.int64)
    labels = np.asarray(is_attack, dtype=np.bool_)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and is_attack must be equally sized one-dimensional arrays")
    benign = np.sort(scores[~labels])[::-1]
    if benign.size == 0:
        raise ValueError("calibration requires at least one benign score")
    allowed = int(np.floor(maximum_false_positive_rate * benign.size))
    if allowed >= benign.size:
        return int(benign[-1]) - 1
    return int(benign[allowed])
