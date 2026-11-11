import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        equal = torch.equal(x, y)
        sin = torch.sin(x)
        return equal, sin


batch_size = 16
width = 4096
height = 2048


def get_inputs():
    x = torch.full((batch_size, height, width), 1)
    y = torch.full((batch_size, height, width), 1)
    return [x, y]


def get_init_inputs():
    return []