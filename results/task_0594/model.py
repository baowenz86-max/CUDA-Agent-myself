import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        x = torch.cat((x1, x2), dim=self.dim)
        equal = torch.equal(x1, x2)
        return x, equal


batch_size = 16
channels = 3
height = 2048
width = 1024


def get_inputs():
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    return [x1, x2]


def get_init_inputs():
    return [1]  # The dimension along which to concatenate