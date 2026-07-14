from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class BinaryLayerTest(unittest.TestCase):
    def test_sign_forward_and_ste_backward(self) -> None:
        import torch

        from bitguard_bnn.models import binary_sign

        values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        output = binary_sign(values)
        self.assertEqual(output.tolist(), [-1.0, -1.0, 1.0, 1.0, 1.0])
        output.sum().backward()
        self.assertEqual(values.grad.tolist(), [0.0, 1.0, 1.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()

