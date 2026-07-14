from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: Tensor) -> Tensor:
        ctx.save_for_backward(values)
        return torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor]:
        (values,) = ctx.saved_tensors
        return (gradient * (values.abs() <= 1).to(gradient.dtype),)


def binary_sign(values: Tensor) -> Tensor:
    return SignSTE.apply(values)


class BinaryActivation(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return binary_sign(values)


class BinaryLinear(nn.Linear):
    """Float master weights, binarized only during the forward pass."""

    def forward(self, values: Tensor) -> Tensor:
        binary_weight = binary_sign(self.weight)
        return F.linear(values, binary_weight, self.bias)


class FeatureGate(nn.Module):
    def __init__(
        self,
        input_groups: np.ndarray,
        group_costs: np.ndarray,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        groups = np.asarray(input_groups, dtype=np.int64)
        costs = np.asarray(group_costs, dtype=np.float32)
        if groups.min(initial=0) < 0 or groups.max(initial=-1) >= len(costs):
            raise ValueError("input group index is outside group_costs")
        self.register_buffer("input_groups", torch.from_numpy(groups))
        self.register_buffer("group_costs", torch.from_numpy(costs / max(float(costs.sum()), 1e-12)))
        self.logits = nn.Parameter(torch.full((len(costs),), 2.0))
        self.temperature = float(temperature)

    def probabilities(self) -> Tensor:
        return torch.sigmoid(self.logits / max(self.temperature, 1e-4))

    def forward(self, values: Tensor) -> Tensor:
        probabilities = self.probabilities()
        hard = (probabilities >= 0.5).to(probabilities.dtype)
        straight_through = hard.detach() - probabilities.detach() + probabilities
        return values * straight_through[self.input_groups]

    def normalized_cost(self) -> Tensor:
        return torch.sum(self.probabilities() * self.group_costs)

    def selected_groups(self) -> Tensor:
        return torch.flatnonzero(self.probabilities() >= 0.5)


class FP32MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class BNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
        binary_first_layer: bool = True,
        feature_gate: FeatureGate | None = None,
    ) -> None:
        super().__init__()
        self.feature_gate = feature_gate
        self.blocks = nn.ModuleList()
        previous = input_dim
        for index, width in enumerate(hidden_dims):
            linear_type: type[nn.Linear] = BinaryLinear if (index > 0 or binary_first_layer) else nn.Linear
            self.blocks.append(
                nn.Sequential(
                    linear_type(previous, width, bias=False),
                    nn.BatchNorm1d(width),
                    BinaryActivation(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            )
            previous = width
        self.output = nn.Linear(previous, output_dim)

    def forward(self, values: Tensor) -> Tensor:
        if self.feature_gate is not None:
            values = self.feature_gate(values)
        for block in self.blocks:
            values = block(values)
        return self.output(values)

    def feature_cost_penalty(self) -> Tensor:
        if self.feature_gate is None:
            return next(self.parameters()).new_zeros(())
        return self.feature_gate.normalized_cost()


@dataclass
class ModelSpec:
    model_type: str
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    dropout: float
    binary_first_layer: bool


def build_model(
    config: dict[str, Any],
    input_dim: int,
    output_dim: int,
    input_groups: np.ndarray,
    feature_costs: np.ndarray,
    *,
    hidden_dims: list[int] | None = None,
    force_bnn: bool = False,
) -> nn.Module:
    cfg = config["model"]
    model_type = "vanilla_bnn" if force_bnn else str(cfg["type"])
    widths = list(hidden_dims if hidden_dims is not None else cfg["hidden_dims"])
    if model_type == "fp32_mlp":
        return FP32MLP(input_dim, widths, output_dim, float(cfg.get("dropout", 0.0)))
    if model_type in {"vanilla_bnn", "cost_aware_bnn"}:
        gate = None
        if model_type == "cost_aware_bnn":
            gate = FeatureGate(
                input_groups,
                feature_costs,
                float(cfg.get("gate_temperature", 1.0)),
            )
        return BNNClassifier(
            input_dim,
            widths,
            output_dim,
            float(cfg.get("dropout", 0.0)),
            bool(cfg.get("binary_first_layer", True)),
            gate,
        )
    raise ValueError(f"{model_type} is not a neural model")


def clamp_binary_master_weights(model: nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BinaryLinear):
                module.weight.clamp_(-1.0, 1.0)


def parameter_summary(model: nn.Module) -> dict[str, int]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    fp32_bytes = parameter_bytes + buffer_bytes
    binary_weights = sum(module.weight.numel() for module in model.modules() if isinstance(module, BinaryLinear))
    packed_binary_bytes = (binary_weights + 7) // 8
    nonbinary_bytes = fp32_bytes - binary_weights * 4
    return {
        "parameters": int(parameters),
        "trainable_parameters": int(trainable),
        "pytorch_tensor_bytes_parameters_and_buffers": int(fp32_bytes),
        "buffer_bytes_included": int(buffer_bytes),
        "binary_weight_count": int(binary_weights),
        "estimated_packed_inference_bytes": int(packed_binary_bytes + max(nonbinary_bytes, 0)),
    }
