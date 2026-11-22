"""Numerically compare selective logits with ordinary masked causal loss."""
import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

from scripts.train_sass_lora import completion_loss


@unittest.skipIf(torch is None, "PyTorch not installed")
class CompletionLossTests(unittest.TestCase):
    def test_matches_dense_loss_and_gradients_with_padding(self):
        torch.manual_seed(3)
        hidden = torch.randn(2, 8, 5)
        weight = torch.randn(5, 11, requires_grad=True)
        labels = torch.tensor([[-100, -100, -100, 3, 4, 5, 6, 7],
                               [-100, -100, 2, 3, 4, -100, -100, -100]])
        dense = hidden @ weight
        reference = F.cross_entropy(dense[:, :-1].reshape(-1, 11),
                                    labels[:, 1:].reshape(-1), ignore_index=-100)
        reference.backward()
        gradient = weight.grad.clone()
        weight.grad = None

        def model(**kwargs):
            positions = kwargs["logits_to_keep"]
            self.assertEqual(positions.tolist(), [1, 2, 3, 4, 5, 6])
            return SimpleNamespace(logits=hidden.index_select(1, positions) @ weight)

        actual = completion_loss(model, {"labels": labels})
        actual.backward()
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(weight.grad, gradient)


if __name__ == "__main__":
    unittest.main()
