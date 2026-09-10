from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import BNNClassifier


class RBCOutput(NamedTuple):
    """Detection and attack-type logits from the shared RBC representation."""

    detection_logits: Tensor
    type_logits: Tensor

    def class_probabilities(self) -> Tensor:
        """Return benign followed by unconditional attack-type probabilities."""

        detection = torch.sigmoid(self.detection_logits)
        if self.type_logits.ndim == self.detection_logits.ndim:
            attack_type_1 = torch.sigmoid(self.type_logits)
            conditional = torch.stack((1.0 - attack_type_1, attack_type_1), dim=-1)
        else:
            conditional = torch.softmax(self.type_logits, dim=-1)
        return torch.cat(
            ((1.0 - detection).unsqueeze(-1), detection.unsqueeze(-1) * conditional),
            dim=-1,
        )


class DetectionFirstBNN(nn.Module):
    """Shared BNN with a primary detector and compact conditional type head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        attack_classes: int = 2,
        dropout: float = 0.0,
        binary_first_layer: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one width")
        if attack_classes < 1:
            raise ValueError("attack_classes must be at least one")
        self.attack_classes = int(attack_classes)
        self.detector = BNNClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout,
            binary_first_layer=binary_first_layer,
        )
        type_outputs = 1 if attack_classes == 2 else attack_classes
        self.type_head = nn.Linear(hidden_dims[-1], type_outputs)
        self._detector_frozen = False

    def hidden_features(self, values: Tensor) -> Tensor:
        """Return the final shared BNN representation used by both heads."""

        if self.detector.feature_gate is not None:
            values = self.detector.feature_gate(values)
        for block in self.detector.blocks:
            values = block(values)
        return values

    def forward(self, values: Tensor) -> RBCOutput:
        shared = self.hidden_features(values)
        type_logits = self.type_head(shared)
        if self.attack_classes == 2:
            type_logits = type_logits.squeeze(-1)
        return RBCOutput(
            self.detector.output(shared).squeeze(-1),
            type_logits,
        )

    def freeze_detector(self) -> None:
        """Freeze shared/detection parameters and BatchNorm statistics."""

        self._detector_frozen = True
        self.detector.requires_grad_(False)
        self.detector.eval()
        self.type_head.requires_grad_(True)

    def freeze_detection(self) -> None:
        """Alias describing the detection-preserving second training stage."""

        self.freeze_detector()

    def enable_batch_norm_recalibration(self) -> None:
        """Update shared BatchNorm statistics without enabling dropout or gradients."""

        if self._detector_frozen:
            raise RuntimeError("BatchNorm recalibration must happen before freezing detection")
        self.eval()
        for module in self.detector.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.reset_running_stats()
                module.momentum = None
                module.train()

    def type_parameters(self):
        return self.type_head.parameters()

    def train(self, mode: bool = True) -> DetectionFirstBNN:
        super().train(mode)
        if self._detector_frozen:
            self.detector.eval()
            self.type_head.train(mode)
        return self


def _normalized_mean(losses: Tensor, sample_weight: Tensor | None) -> Tensor:
    if sample_weight is None:
        return losses.mean()
    weights = sample_weight.reshape(-1).to(device=losses.device, dtype=losses.dtype)
    if weights.numel() != losses.numel():
        raise ValueError("sample_weight must match the number of logits")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("sample_weight must be finite and non-negative")
    denominator = weights.sum()
    if float(denominator.detach()) <= 0.0:
        raise ValueError("sample_weight must have a positive sum")
    return (losses * weights).sum() / denominator


def _group_normalized_means(
    losses: Tensor,
    group_ids: Tensor,
    sample_weight: Tensor | None,
) -> Tensor:
    """Compute every observed group's normalized mean in one pass."""

    unique_groups, inverse = torch.unique(group_ids, return_inverse=True)
    group_count = unique_groups.numel()
    if sample_weight is None:
        numerators = losses.new_zeros(group_count).index_add_(0, inverse, losses)
        denominators = torch.bincount(inverse, minlength=group_count).to(dtype=losses.dtype)
    else:
        weighted_losses = losses * sample_weight
        numerators = losses.new_zeros(group_count).index_add_(0, inverse, weighted_losses)
        denominators = losses.new_zeros(group_count).index_add_(0, inverse, sample_weight)
        if bool((denominators <= 0).any()):
            raise ValueError("sample_weight must have a positive sum")
    means = numerators / denominators
    if unique_groups.is_floating_point() and bool(torch.isnan(unique_groups).any()):
        means = means.masked_fill(torch.isnan(unique_groups), torch.nan)
    return means


def normalized_detection_loss(
    logits: Tensor,
    target: Tensor,
    *,
    sample_weight: Tensor | None = None,
    group_ids: Tensor | None = None,
    worst_group_blend: float = 0.0,
    teacher_logits: Tensor | None = None,
    distillation_alpha: float = 0.0,
    temperature: float = 2.0,
) -> Tensor:
    """Normalized binary loss with optional worst-group and teacher blending."""

    if not 0.0 <= worst_group_blend <= 1.0:
        raise ValueError("worst_group_blend must be between 0 and 1")
    if not 0.0 <= distillation_alpha <= 1.0:
        raise ValueError("distillation_alpha must be between 0 and 1")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    flat_logits = logits.reshape(-1)
    flat_target = target.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if flat_logits.numel() == 0 or flat_target.numel() != flat_logits.numel():
        raise ValueError("target must match a non-empty logits tensor")
    if bool(((flat_target != 0) & (flat_target != 1)).any()):
        raise ValueError("target must contain only binary labels")

    per_sample = F.binary_cross_entropy_with_logits(flat_logits, flat_target, reduction="none")
    supervised = _normalized_mean(per_sample, sample_weight)

    if worst_group_blend > 0.0:
        if group_ids is None:
            raise ValueError("group_ids are required when worst_group_blend is positive")
        flat_groups = group_ids.reshape(-1).to(device=logits.device)
        if flat_groups.numel() != flat_logits.numel():
            raise ValueError("group_ids must match the number of logits")
        flat_weights = (
            None
            if sample_weight is None
            else sample_weight.reshape(-1).to(device=logits.device, dtype=logits.dtype)
        )
        group_means = _group_normalized_means(per_sample, flat_groups, flat_weights)
        worst_group = group_means.max()
        supervised = (1.0 - worst_group_blend) * supervised + worst_group_blend * worst_group

    if teacher_logits is None or distillation_alpha == 0.0:
        return supervised
    flat_teacher = teacher_logits.detach().reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if flat_teacher.numel() != flat_logits.numel():
        raise ValueError("teacher_logits must match the number of logits")
    soft_target = torch.sigmoid(flat_teacher / temperature)
    per_sample_distillation = F.binary_cross_entropy_with_logits(
        flat_logits / temperature,
        soft_target,
        reduction="none",
    ) * (temperature**2)
    distillation = _normalized_mean(per_sample_distillation, sample_weight)
    return (1.0 - distillation_alpha) * supervised + distillation_alpha * distillation


def conditional_type_loss(
    type_logits: Tensor,
    attack_type_target: Tensor,
    *,
    attack_mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Normalized binary type loss evaluated only for detected attack classes."""

    binary = type_logits.ndim == 1
    rows = type_logits.shape[0]
    flat_target = attack_type_target.reshape(-1).to(device=type_logits.device)
    if flat_target.numel() != rows:
        raise ValueError("attack_type_target must match the number of type-logit rows")
    if attack_mask is None:
        selected = torch.ones(rows, device=type_logits.device, dtype=torch.bool)
    else:
        selected = attack_mask.reshape(-1).to(device=type_logits.device, dtype=torch.bool)
        if selected.numel() != rows:
            raise ValueError("attack_mask must match the number of type-logit rows")
    if not bool(selected.any()):
        return type_logits.sum() * 0.0
    selected_target = flat_target[selected]
    if binary:
        if bool(((selected_target != 0) & (selected_target != 1)).any()):
            raise ValueError("selected attack_type_target values must be binary")
        losses = F.binary_cross_entropy_with_logits(
            type_logits.reshape(rows)[selected],
            selected_target.to(dtype=type_logits.dtype),
            reduction="none",
        )
    else:
        losses = F.cross_entropy(
            type_logits[selected], selected_target.to(dtype=torch.long), reduction="none"
        )
    selected_weight = None
    if sample_weight is not None:
        flat_weight = sample_weight.reshape(-1).to(
            device=type_logits.device, dtype=type_logits.dtype
        )
        if flat_weight.numel() != rows:
            raise ValueError("sample_weight must match the number of type-logit rows")
        selected_weight = flat_weight[selected]
    return _normalized_mean(losses, selected_weight)
