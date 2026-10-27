import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = self.upsample(x)
        x = torch.squeeze(x)
        return x


batch_size = 16
scale_factor = 2.0
mode = "bilinear"
width = 256
height = 512  # Changed from 256 to 512 to ensure at least one dimension >=512


def get_inputs():
    x = torch.randn(batch_size, 32, height, width)  # Increased channels from 1 to 32
    return [x]


def get_init_inputs():
    return [scale_factor, mode]