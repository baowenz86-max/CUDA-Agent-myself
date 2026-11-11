import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x, y):
        x = x * y
        return x


batch_size = 512
in_channels = 3
height, width = 512, 512
scaling_factor = 2.0


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    y = torch.randn(batch_size, in_channels, height, width)
    return [x, y]


def get_init_inputs():
    input_shape = (batch_size, in_channels, height, width)
    return [input_shape, scaling_factor]