import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x):
        x_max, _ = torch.max(x, dim=self.input_shape)
        x_cumsum = torch.cumsum(x, dim=self.input_shape)
        return x_max, x_cumsum


input_height = 8192
input_width = 128


def get_inputs():
    x = torch.randn(input_height, input_width)
    return [x]


def get_init_inputs():
    return [0]