import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels: int, size: int, alpha: float, beta: float, k: float):
        super().__init__()
        self.local_response_norm = nn.LocalResponseNorm(size, alpha, beta, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local_response_norm(x)


batch_size = 16
in_channels = 3
height = 1024
width = 512


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [3, 5, 1e-4, 0.75, 1.0]