import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x)
        x = nn.LogSoftmax()(x)
        return x


batch_size = 1024
width = 1024
height = 256


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []