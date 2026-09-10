from __future__ import annotations

import unittest

import torch
from torch import Tensor
from torch.nn import functional as F

from bitguard_bnn.rbc_model import normalized_detection_loss


def _legacy_normalized_mean(losses: Tensor, sample_weight: Tensor | None) -> Tensor:
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


def _legacy_loss(
    logits: Tensor,
    target: Tensor,
    *,
    sample_weight: Tensor | None = None,
    group_ids: Tensor | None = None,
    worst_group_blend: float = 0.0,
) -> Tensor:
    flat_logits = logits.reshape(-1)
    flat_target = target.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if flat_logits.numel() == 0 or flat_target.numel() != flat_logits.numel():
        raise ValueError("target must match a non-empty logits tensor")
    if bool(((flat_target != 0) & (flat_target != 1)).any()):
        raise ValueError("target must contain only binary labels")
    per_sample = F.binary_cross_entropy_with_logits(flat_logits, flat_target, reduction="none")
    supervised = _legacy_normalized_mean(per_sample, sample_weight)
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
        group_means = [
            _legacy_normalized_mean(
                per_sample[flat_groups == group],
                None if flat_weights is None else flat_weights[flat_groups == group],
            )
            for group in torch.unique(flat_groups)
        ]
        worst_group = torch.stack(group_means).max()
        supervised = (1.0 - worst_group_blend) * supervised + worst_group_blend * worst_group
    return supervised


class RBCGroupLossOptimizationTest(unittest.TestCase):
    def _assert_value_and_gradient_match(
        self, logits: Tensor, target: Tensor, groups: Tensor, weights: Tensor | None
    ) -> None:
        legacy_logits = logits.detach().clone().requires_grad_(True)
        optimized_logits = logits.detach().clone().requires_grad_(True)
        legacy = _legacy_loss(
            legacy_logits,
            target,
            sample_weight=weights,
            group_ids=groups,
            worst_group_blend=0.73,
        )
        optimized = normalized_detection_loss(
            optimized_logits,
            target,
            sample_weight=weights,
            group_ids=groups,
            worst_group_blend=0.73,
        )
        legacy.backward()
        optimized.backward()
        torch.testing.assert_close(optimized, legacy, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            optimized_logits.grad, legacy_logits.grad, rtol=1e-6, atol=1e-7
        )

    def test_randomized_weighted_and_unweighted_values_and_gradients_match(self) -> None:
        generator = torch.Generator().manual_seed(20260909)
        for weighted in (False, True):
            for sample_count, group_count in ((37, 7), (257, 31)):
                logits = torch.randn(sample_count, generator=generator)
                target = torch.randint(0, 2, (sample_count,), generator=generator).float()
                groups = torch.randint(0, group_count, (sample_count,), generator=generator)
                weights = (
                    torch.rand(sample_count, generator=generator) + 0.05 if weighted else None
                )
                self._assert_value_and_gradient_match(logits, target, groups, weights)

    def test_tied_worst_groups_preserve_max_gradient_distribution(self) -> None:
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        groups = torch.tensor([10, 10, 20, 20])
        weights = torch.tensor([1.0, 3.0, 1.0, 3.0])
        self._assert_value_and_gradient_match(logits, target, groups, weights)

    def test_group_validation_behavior_is_preserved(self) -> None:
        logits = torch.tensor([0.0, 1.0])
        target = torch.tensor([0.0, 1.0])
        cases = (
            ({"worst_group_blend": 1.0}, "group_ids are required"),
            ({"group_ids": torch.tensor([0]), "worst_group_blend": 1.0}, "group_ids must match"),
            (
                {
                    "sample_weight": torch.tensor([0.0, 1.0]),
                    "group_ids": torch.tensor([0, 1]),
                    "worst_group_blend": 1.0,
                },
                "sample_weight must have a positive sum",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(message=message):
                for implementation in (_legacy_loss, normalized_detection_loss):
                    with self.assertRaisesRegex(ValueError, message):
                        implementation(logits, target, **kwargs)

    def test_nan_group_id_preserves_legacy_nan_result(self) -> None:
        logits = torch.tensor([0.0, 1.0])
        target = torch.tensor([0.0, 1.0])
        groups = torch.tensor([0.0, float("nan")])
        actual = normalized_detection_loss(
            logits, target, group_ids=groups, worst_group_blend=1.0
        )
        self.assertTrue(torch.isnan(actual))


if __name__ == "__main__":
    unittest.main()
