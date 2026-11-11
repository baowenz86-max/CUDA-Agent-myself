import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple:
        max_value = torch.max(x)
        std_value = torch.std(x)
        return max_value, std_value


batch_size = 16
in_channels = 3
height = 2560
width = 2560


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [3]