import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding = nn.ZeroPad2d(padding_size)

    def forward(self, x):
        return self.padding(x)


def get_inputs():
    batch_size = 16
    in_channels = 3
    height = 2048  # Increased from 512 to increase computation
    width = 2048   # Increased from 256 to increase computation
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [(1, 2, 1, 2)]  # padding size as a tuple (top, bottom, left, right)