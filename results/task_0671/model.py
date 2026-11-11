import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x)
        x = torch.floor(x)
        return x


batch_size = 16
channels = 3
height = 2048
width = 2560


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [get_inputs()[0].size()]