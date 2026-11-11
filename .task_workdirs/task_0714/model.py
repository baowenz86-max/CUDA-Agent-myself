import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = torch.abs(x)
        x = nn.Softplus()(x)
        return x


batch_size = 16
channels = 3
height = 2048  # Increased from 512 to increase computation
width = 2048   # Increased from 256 to increase computation


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [get_inputs()[0].size(1)]