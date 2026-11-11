import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.pixel_shuffle(x, self.scale_factor)


batch_size = 16
channels = 64
height = 1024  # Increased from 512
width = 256    # Increased from 128
scale_factor = 2


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels, scale_factor]