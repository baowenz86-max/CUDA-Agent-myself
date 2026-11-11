import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.upsample_nearest(x, scale_factor=self.scale_factor)
        x = F.relu(x)
        return x


batch_size = 512
height, width = 256, 256
in_channels = 3
scale_factor = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [scale_factor, "nearest"]