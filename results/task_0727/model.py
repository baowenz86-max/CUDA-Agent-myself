import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, start, end, step, scale_factor):
        super().__init__()
        self.range = torch.arange(start, end, step)
        self.upsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x):
        x = self.upsample(x)
        return x


batch_size = 512
channels = 3
height, width = 256, 256  # Increased from 64x64 to 256x256
scale_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [0, 10, 1, scale_factor]