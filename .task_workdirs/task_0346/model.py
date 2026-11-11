import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        return self.upsample(x)


batch_size = 16
channels = 3
height = 1024
width = 1024
scale_factor = 2


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [scale_factor]