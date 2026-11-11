import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, size: int, alpha: float, beta: float, k: float):
        super().__init__()
        self.cross_map_lrn = nn.CrossMapLRN2d(size, alpha, beta, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cross_map_lrn(x)


batch_size = 512
channels = 16
height = 64
width = 64
size = 5
alpha = 1e-4
beta = 0.75
k = 1.0


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [size, alpha, beta, k]