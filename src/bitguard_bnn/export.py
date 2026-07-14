from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, save_json
from .preprocess import FeaturePreprocessor


def fold_linear_batchnorm(linear: Any, batchnorm: Any) -> dict[str, np.ndarray]:
    """Fold BN + sign into a dot-product threshold and polarity."""

    weight = linear.weight.detach().cpu().numpy().astype(np.float32)
    bias = (
        linear.bias.detach().cpu().numpy().astype(np.float32)
        if linear.bias is not None
        else np.zeros(weight.shape[0], dtype=np.float32)
    )
    mean = batchnorm.running_mean.detach().cpu().numpy().astype(np.float32)
    variance = batchnorm.running_var.detach().cpu().numpy().astype(np.float32)
    gamma = batchnorm.weight.detach().cpu().numpy().astype(np.float32)
    beta = batchnorm.bias.detach().cpu().numpy().astype(np.float32)
    scale = gamma / np.sqrt(variance + float(batchnorm.eps))
    threshold = np.empty_like(scale)
    polarity = np.where(scale >= 0, 1, -1).astype(np.int8)
    constant = np.zeros_like(polarity, dtype=np.int8)
    regular = np.abs(scale) > 1e-12
    threshold[regular] = mean[regular] - beta[regular] / scale[regular] - bias[regular]
    threshold[~regular] = 0.0
    constant[~regular] = np.where(beta[~regular] >= 0, 1, -1)
    return {
        "weight": weight,
        "threshold": threshold.astype(np.float32),
        "polarity": polarity,
        "constant_output": constant,
    }


def folded_sign(dot_product: np.ndarray, folded: dict[str, np.ndarray]) -> np.ndarray:
    threshold = folded["threshold"][None, :]
    polarity = folded["polarity"][None, :]
    positive = np.where(polarity > 0, dot_product >= threshold, dot_product <= threshold)
    output = np.where(positive, 1.0, -1.0).astype(np.float32)
    constant = folded["constant_output"]
    if np.any(constant):
        output[:, constant != 0] = constant[constant != 0]
    return output


def _build_from_checkpoint(config: dict[str, Any], checkpoint: dict[str, Any], preprocessor: FeaturePreprocessor) -> Any:
    from .models import build_model

    config = dict(config)
    config["model"] = dict(config["model"])
    config["model"]["type"] = checkpoint["model_type"]
    config["model"]["hidden_dims"] = checkpoint["hidden_dims"]
    config["model"]["dropout"] = checkpoint["dropout"]
    config["model"]["binary_first_layer"] = checkpoint["binary_first_layer"]
    input_groups = np.asarray(checkpoint["input_groups"], dtype=np.int64)
    costs = np.asarray(checkpoint.get("feature_costs", preprocessor.feature_costs), dtype=np.float32)
    model = build_model(
        config,
        int(checkpoint["input_dim"]),
        int(checkpoint["output_dim"]),
        input_groups,
        costs,
        hidden_dims=list(checkpoint["hidden_dims"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def export_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    import torch

    from .models import BinaryLinear

    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("edge export requires a neural best_model.pt checkpoint")
    config = load_config(run_dir / "resolved_config.yaml")
    preprocessor = FeaturePreprocessor.load(run_dir / "preprocessor.joblib")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _build_from_checkpoint(config, checkpoint, preprocessor)
    if not hasattr(model, "blocks"):
        raise ValueError("packed export currently supports BNNClassifier, not FP32MLP")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
        "format_version": 1,
        "bit_order": "little",
        "input_encoder": preprocessor.encoder.kind,
        "encoder_bits": preprocessor.encoder.bits,
        "selected_features": preprocessor.selected_features,
        "output_labels": checkpoint["output_labels"],
        "layers": [],
        "arithmetic": "See each layer's operation field; hybrid/none first inputs are not XNOR eligible.",
        "automatic_network_action": False,
    }
    active_encoded = np.arange(preprocessor.encoded_dimension, dtype=np.int64)
    active_groups = np.arange(len(preprocessor.selected_features), dtype=np.int64)
    if getattr(model, "feature_gate", None) is not None:
        probabilities = model.feature_gate.probabilities().detach().cpu().numpy()
        arrays["feature_gate_probability"] = probabilities.astype(np.float32)
        arrays["feature_gate_hard_mask"] = (probabilities >= 0.5).astype(np.uint8)
        manifest["feature_gate_scope_note"] = (
            "The classifier applies the hard mask, but the open-set benign-distance detector uses "
            "all selected features. Acquisition-cost claims must include those anomaly features."
        )
    arrays["active_original_feature_indices"] = active_groups
    arrays["active_encoded_indices"] = active_encoded
    assert preprocessor.selected_indices is not None
    arrays["imputer_median"] = preprocessor.imputer.statistics_[
        preprocessor.selected_indices[active_groups]
    ].astype(np.float32)
    scaler_center = getattr(preprocessor.scaler, "center_", None)
    if scaler_center is None:
        scaler_center = getattr(preprocessor.scaler, "mean_", np.zeros(len(preprocessor.selected_features)))
    scaler_scale = getattr(preprocessor.scaler, "scale_", np.ones(len(preprocessor.selected_features)))
    arrays["scaler_center"] = np.asarray(scaler_center, dtype=np.float32)[active_groups]
    arrays["scaler_scale"] = np.asarray(scaler_scale, dtype=np.float32)[active_groups]
    arrays["open_set_benign_center"] = np.asarray(preprocessor.benign_center, dtype=np.float32)[active_groups]
    arrays["open_set_distance_threshold"] = np.asarray(
        [preprocessor.open_distance_threshold], dtype=np.float32
    )
    if preprocessor.encoder.thresholds is not None:
        thresholds = np.asarray(preprocessor.encoder.thresholds, dtype=np.float32)
        arrays["encoder_thresholds"] = thresholds[active_groups]
    manifest["preprocessing"] = {
        "raw_feature_order": [preprocessor.selected_features[index] for index in active_groups],
        "imputation": "median",
        "scaling": type(preprocessor.scaler).__name__,
        "open_set_confidence_threshold": float(
            preprocessor.config["preprocess"]["open_set"]["confidence_threshold"]
        ),
        "open_set_distance": "mean absolute distance in scaled selected-feature space",
    }
    manifest["temporal"] = config["temporal"]
    manifest["cascade_calibration_file"] = (
        "cascade_calibration.json" if (run_dir / "cascade_calibration.json").exists() else None
    )
    manifest["scope_note"] = (
        "Main BNN is bit-packed. A Tiny checkpoint/calibration is bundled when available, but a "
        "target-specific Tiny packed runtime is still required for a complete cascade deployment."
    )
    scaled_reference = np.random.default_rng(2309).normal(
        size=(64, len(preprocessor.selected_features))
    ).astype(np.float32)
    reference = preprocessor.encoder.transform(scaled_reference)
    with torch.inference_mode():
        full_model_logits = model(torch.from_numpy(reference)).numpy()
        if getattr(model, "feature_gate", None) is not None:
            model_reference = model.feature_gate(torch.from_numpy(reference)).numpy()
        else:
            model_reference = reference
    exported_reference = model_reference.copy()
    parity: list[bool] = []
    for index, block in enumerate(model.blocks):
        linear = block[0]
        batchnorm = block[1]
        folded = fold_linear_batchnorm(linear, batchnorm)
        layer_weight = folded["weight"]
        if index == 0:
            model_input = model_reference
            exported_input = exported_reference
        else:
            model_input = model_reference
            exported_input = exported_reference
        if isinstance(linear, BinaryLinear):
            signed_weight = np.where(layer_weight >= 0, 1.0, -1.0).astype(np.float32)
            arrays[f"layer_{index}_weight_bits"] = np.packbits(
                signed_weight > 0, axis=1, bitorder="little"
            )
            arrays[f"layer_{index}_input_bits"] = np.asarray([signed_weight.shape[1]], dtype=np.int32)
            storage = "packed_binary"
            exported_dot = exported_input @ signed_weight.T
            xnor_eligible = index > 0 or preprocessor.encoder.kind in {"sign", "thermometer"}
            if xnor_eligible:
                matches = (exported_input[:, None, :] > 0) == (signed_weight[None, :, :] > 0)
                xnor_dot = 2.0 * matches.sum(axis=2) - signed_weight.shape[1]
                xnor_parity = bool(np.array_equal(xnor_dot.astype(np.float32), exported_dot))
                operation = "XNOR_popcount_threshold"
            else:
                xnor_parity = None
                operation = "low_bit_activation_times_binary_weight_accumulate"
        else:
            arrays[f"layer_{index}_weight_fp32"] = layer_weight.astype(np.float32)
            storage = "fp32"
            exported_dot = exported_input @ layer_weight.T
            xnor_parity = None
            operation = "fp32_matrix_vector_threshold"
        arrays[f"layer_{index}_threshold"] = folded["threshold"]
        arrays[f"layer_{index}_polarity"] = folded["polarity"]
        arrays[f"layer_{index}_constant_output"] = folded["constant_output"]
        exported_reference = folded_sign(exported_dot, folded)
        with torch.inference_mode():
            model_reference = block(torch.from_numpy(model_input)).numpy()
        layer_parity = bool(np.array_equal(exported_reference, model_reference))
        parity.append(layer_parity)
        manifest["layers"].append(
            {
                "index": index,
                "input_dimension": int(layer_weight.shape[1]),
                "output_dimension": int(layer_weight.shape[0]),
                "weight_storage": storage,
                "input_representation": (
                    preprocessor.encoder.kind if index == 0 else "binary_activation"
                ),
                "operation": operation,
                "xnor_dot_parity_passed": xnor_parity,
                "batchnorm_folded": True,
                "parity_passed": layer_parity,
            }
        )
    arrays["output_weight_fp32"] = model.output.weight.detach().cpu().numpy().astype(np.float32)
    arrays["output_bias_fp32"] = model.output.bias.detach().cpu().numpy().astype(np.float32)
    exported_logits = (
        exported_reference @ arrays["output_weight_fp32"].T + arrays["output_bias_fp32"]
    )
    logits_parity = bool(np.allclose(exported_logits, full_model_logits, atol=1e-5, rtol=1e-5))
    np.savez_compressed(output_dir / "bitguard_edge_weights.npz", **arrays)
    manifest["folding_parity_passed"] = bool(all(parity))
    manifest["end_to_end_logit_parity_passed"] = logits_parity
    manifest["files"] = ["bitguard_edge_weights.npz", "bitguard_edge_manifest.json"]
    for optional_name in (
        "cascade_calibration.json",
        "boolean_fast_path.json",
        "tiny_model.pt",
    ):
        source = run_dir / optional_name
        if source.exists():
            shutil.copy2(source, output_dir / optional_name)
            manifest["files"].append(optional_name)
    save_json(manifest, output_dir / "bitguard_edge_manifest.json")
    if not all(parity) or not logits_parity:
        raise RuntimeError("edge export parity failed; files were written for diagnosis only")
    return {
        "output_dir": str(output_dir),
        "weights": str(output_dir / "bitguard_edge_weights.npz"),
        "manifest": str(output_dir / "bitguard_edge_manifest.json"),
        "folding_parity_passed": True,
        "end_to_end_logit_parity_passed": True,
    }
