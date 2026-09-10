from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def feature_penalty_coefficient(
    loss_config: dict[str, Any],
    optimizer_step: int,
    total_optimizer_steps: int,
) -> float:
    """Return the scheduled feature-cost weight for a zero-based optimizer step."""

    target = float(loss_config.get("lambda_feature", 0.0))
    if target == 0.0:
        return 0.0
    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    warmup = float(loss_config.get("feature_penalty_warmup_fraction", 0.0))
    ramp = float(loss_config.get("feature_penalty_ramp_fraction", 0.0))
    warmup_steps = warmup * total_optimizer_steps
    if warmup_steps > 0.0 and optimizer_step <= warmup_steps:
        return 0.0
    if ramp <= 0.0:
        return target
    progress = (optimizer_step - warmup_steps) / (ramp * total_optimizer_steps)
    return target * min(max(progress, 0.0), 1.0)


class FocalLoss(nn.Module):
    def __init__(self, weight: Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = float(gamma)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        selected_log = log_probabilities.gather(1, target[:, None]).squeeze(1)
        selected_probability = probabilities.gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - selected_probability) ** self.gamma) * selected_log
        if self.weight is not None:
            weights = self.weight[target]
            return (loss * weights).sum() / weights.sum().clamp_min(1e-12)
        return loss.mean()


@dataclass
class LossOutput:
    total: Tensor
    detection: Tensor
    feature_cost: Tensor
    false_negative: Tensor
    false_positive: Tensor
    distillation: Tensor


class BitGuardObjective(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        class_weights: Tensor | None,
        benign_index: int,
    ) -> None:
        super().__init__()
        cfg = config["loss"]
        if cfg["type"] == "focal":
            self.detection_loss: nn.Module = FocalLoss(class_weights, float(cfg["focal_gamma"]))
        elif cfg["type"] in {"weighted_ce", "cross_entropy"}:
            self.detection_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"unsupported loss.type: {cfg['type']}")
        self.lambda_feature = float(cfg.get("lambda_feature", 0.0))
        self.beta_fn = float(cfg.get("beta_fn", 0.0))
        self.gamma_fp = float(cfg.get("gamma_fp", 0.0))
        self.distillation_alpha = float(cfg.get("distillation_alpha", 0.0))
        self.temperature = float(cfg.get("distillation_temperature", 2.0))
        self.benign_index = int(benign_index)

    def forward(
        self,
        model: nn.Module,
        logits: Tensor,
        target: Tensor,
        teacher_logits: Tensor | None = None,
        feature_coefficient: float | None = None,
    ) -> LossOutput:
        detection = self.detection_loss(logits, target)
        probabilities = torch.softmax(logits, dim=1)
        benign_probability = probabilities[:, self.benign_index]
        is_attack = (target != self.benign_index).to(probabilities.dtype)
        false_negative = (is_attack * benign_probability).sum() / is_attack.sum().clamp_min(1.0)
        is_benign = 1.0 - is_attack
        false_positive = (is_benign * (1.0 - benign_probability)).sum() / is_benign.sum().clamp_min(1.0)
        if hasattr(model, "feature_cost_penalty"):
            feature_cost = model.feature_cost_penalty()
        else:
            feature_cost = logits.new_zeros(())
        distillation = logits.new_zeros(())
        if teacher_logits is not None and self.distillation_alpha > 0:
            temperature = self.temperature
            distillation = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits.detach() / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature**2)
        total = (
            (1.0 - self.distillation_alpha) * detection
            + self.distillation_alpha * distillation
            + (
                self.lambda_feature
                if feature_coefficient is None
                else float(feature_coefficient)
            )
            * feature_cost
            + self.beta_fn * false_negative
            + self.gamma_fp * false_positive
        )
        return LossOutput(total, detection, feature_cost, false_negative, false_positive, distillation)

