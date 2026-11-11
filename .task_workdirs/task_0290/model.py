import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_channels, tril_offset):
        super().__init__()
        self.tril_offset = tril_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tril(x, self.tril_offset)


batch_size = 16
n_channels = 3
height = 2048
width = 2048
tril_offset = 1


def get_inputs():
    x = torch.randn(batch_size, n_channels, height, width)
    return [x]


def get_init_inputs():
    return [n_channels, tril_offset]