import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(out_channels)

    def forward(self, x):
        return self.adaptive_avg_pool3d(x)


batch_size = 512
in_channels = 3
depth, height, width = 32, 32, 32


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [3, 2]