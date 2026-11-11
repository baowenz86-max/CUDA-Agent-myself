import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x1, x2):
        x = torch.logical_or(x1, x2)
        x = torch.round(x.float())
        return x


batch_size = 512
height, width = 512, 512  # Increased from 128x128 to 512x512
input_shape = (batch_size, 1, height, width)


def get_inputs():
    x1 = torch.randn(input_shape) > 0
    x2 = torch.randn(input_shape) > 0
    return [x1, x2]


def get_init_inputs():
    return [input_shape]