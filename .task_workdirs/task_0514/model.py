import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        mean = torch.mean(x, dim=1)  # Assume we want to calculate the mean over the dimension 1
        return mean


batch_size = 16
sequence_length = 16384  # Increased from 256 to increase computation
hidden_size = 512


def get_inputs():
    x = torch.randn(batch_size, sequence_length, hidden_size)
    return [x]


def get_init_inputs():
    return [hidden_size]  # normalized_shape for LayerNorm