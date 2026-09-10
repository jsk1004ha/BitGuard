from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class DetectionFirstBNNTest(unittest.TestCase):
    def test_compact_outputs_form_three_class_probabilities(self) -> None:
        import torch

        from bitguard_bnn.rbc_model import DetectionFirstBNN

        model = DetectionFirstBNN(4, [6, 3], binary_first_layer=False).eval()
        output = model(torch.randn(5, 4))

        self.assertEqual(output.detection_logits.shape, (5,))
        self.assertEqual(output.type_logits.shape, (5,))
        probabilities = output.class_probabilities()
        self.assertEqual(probabilities.shape, (5, 3))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(5))

    def test_general_type_head_uses_one_logit_per_attack_class(self) -> None:
        import torch

        from bitguard_bnn.rbc_model import DetectionFirstBNN, conditional_type_loss

        model = DetectionFirstBNN(4, [5], attack_classes=4, binary_first_layer=False).eval()
        output = model(torch.randn(3, 4))
        self.assertEqual(output.type_logits.shape, (3, 4))
        self.assertEqual(output.class_probabilities().shape, (3, 5))
        loss = conditional_type_loss(
            output.type_logits, torch.tensor([0, 3, 2]), attack_mask=torch.tensor([1, 1, 0])
        )
        self.assertTrue(torch.isfinite(loss))

    def test_single_attack_type_is_a_non_compact_one_class_head(self) -> None:
        import torch

        from bitguard_bnn.rbc_model import DetectionFirstBNN, conditional_type_loss

        model = DetectionFirstBNN(4, [5], attack_classes=1, binary_first_layer=False).eval()
        output = model(torch.randn(3, 4))
        self.assertEqual(output.type_logits.shape, (3, 1))
        probabilities = output.class_probabilities()
        self.assertEqual(probabilities.shape, (3, 2))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3))
        torch.testing.assert_close(
            conditional_type_loss(output.type_logits, torch.zeros(3, dtype=torch.long)),
            torch.tensor(0.0),
        )

        model.freeze_detector()
        model.train()
        self.assertFalse(model.detector.training)
        self.assertTrue(model.type_head.training)

    def test_type_training_preserves_frozen_detection_and_batch_norm(self) -> None:
        import torch

        from bitguard_bnn.rbc_model import DetectionFirstBNN

        torch.manual_seed(7)
        model = DetectionFirstBNN(4, [5, 3], dropout=0.5, binary_first_layer=False)
        model.freeze_detector()
        model.train()
        values = torch.randn(8, 4)
        before = model(values).detection_logits.detach().clone()
        running_means = [block[1].running_mean.clone() for block in model.detector.blocks]

        optimizer = torch.optim.SGD(model.type_parameters(), lr=0.2)
        optimizer.zero_grad()
        output = model(values)
        torch.nn.functional.binary_cross_entropy_with_logits(
            output.type_logits, torch.randint(0, 2, (8,), dtype=torch.float32)
        ).backward()
        optimizer.step()

        after = model(values).detection_logits.detach()
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        for block, running_mean in zip(model.detector.blocks, running_means, strict=True):
            torch.testing.assert_close(block[1].running_mean, running_mean, rtol=0, atol=0)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.detector.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.type_head.parameters()))

    def test_batch_norm_recalibration_updates_only_running_statistics(self) -> None:
        import torch

        from bitguard_bnn.rbc_model import DetectionFirstBNN

        model = DetectionFirstBNN(4, [5], dropout=0.5, binary_first_layer=False)
        batch_norm = model.detector.blocks[0][1]
        batch_norm.running_mean.fill_(7.0)
        model.enable_batch_norm_recalibration()
        before = batch_norm.running_mean.clone()
        torch.testing.assert_close(before, torch.zeros_like(before), rtol=0, atol=0)
        self.assertIsNone(batch_norm.momentum)
        with torch.no_grad():
            model(torch.full((8, 4), 3.0))
        self.assertFalse(torch.equal(batch_norm.running_mean, before))
        self.assertFalse(model.detector.blocks[0][3].training)

        model.freeze_detector()
        with self.assertRaisesRegex(RuntimeError, "before freezing"):
            model.enable_batch_norm_recalibration()


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class NormalizedDetectionLossTest(unittest.TestCase):
    def test_weight_scale_does_not_change_normalized_loss(self) -> None:
        import torch

        from bitguard_bnn.rbc_model import normalized_detection_loss

        logits = torch.tensor([-1.0, 0.5, 2.0])
        target = torch.tensor([0.0, 1.0, 0.0])
        weights = torch.tensor([1.0, 2.0, 4.0])

        original = normalized_detection_loss(logits, target, sample_weight=weights)
        scaled = normalized_detection_loss(logits, target, sample_weight=weights * 100)
        torch.testing.assert_close(original, scaled)

    def test_worst_group_and_teacher_terms_are_bounded_blends(self) -> None:
        import torch

        from bitguard_bnn.rbc_model import normalized_detection_loss

        logits = torch.tensor([-2.0, -2.0, -2.0, 2.0])
        target = torch.tensor([0.0, 0.0, 1.0, 1.0])
        groups = torch.tensor([0, 0, 1, 1])
        average = normalized_detection_loss(logits, target)
        worst = normalized_detection_loss(
            logits, target, group_ids=groups, worst_group_blend=1.0
        )
        blended = normalized_detection_loss(
            logits, target, group_ids=groups, worst_group_blend=0.25
        )
        torch.testing.assert_close(blended, 0.75 * average + 0.25 * worst)

        teacher = torch.tensor([2.0, 2.0, 2.0, -2.0])
        distilled = normalized_detection_loss(
            logits,
            target,
            teacher_logits=teacher,
            distillation_alpha=1.0,
            temperature=2.0,
        )
        mixed = normalized_detection_loss(
            logits,
            target,
            teacher_logits=teacher,
            distillation_alpha=0.4,
            temperature=2.0,
        )
        torch.testing.assert_close(mixed, 0.6 * average + 0.4 * distilled)

    def test_type_loss_uses_attack_rows_only(self) -> None:
        import torch

        from bitguard_bnn.rbc_model import conditional_type_loss

        logits = torch.tensor([-1.0, 0.0, 2.0])
        target = torch.tensor([99.0, 0.0, 1.0])
        attack_mask = torch.tensor([False, True, True])
        actual = conditional_type_loss(logits, target, attack_mask=attack_mask)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[attack_mask], target[attack_mask]
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
