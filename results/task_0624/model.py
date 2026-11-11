import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x):
        x = torch.sum(x, dim=1)  # sum along the channel dimension
        x = torch.exp(x)  # apply exponential function
        return x


batch_size = 512
height = 512
width = 512
in_channels = 3
out_channels = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels]