import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, scale_factor, mode, pools):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.pools = nn.ParameterList([nn.AdaptiveAvgPool2d(pool) for pool in pools])

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        pooled_tensors = [pool(x) for pool in self.pools]
        return pooled_tensors


batch_size = 1024  # Increased from 512 to double the computation
in_channels = 3
height, width = 64, 64
scale_factor = 2
mode = "bilinear"
pools = [(8, 8), (4, 4)]


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [scale_factor, mode, pools]