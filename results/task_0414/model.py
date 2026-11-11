import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


batch_size = 16
channels = 3
height = 512  # Increased from 64 to increase computation
width = 1024  # Increased from 512 to increase computation
scale_factor = 2.0
mode = "bicubic"  # Changed from nearest to more computationally intensive mode


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [scale_factor, mode]