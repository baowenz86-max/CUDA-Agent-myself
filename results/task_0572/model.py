import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, pool_size):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, x):
        x = torch.abs(x)
        x = self.adaptive_avg_pool(x)
        return x


batch_size = 16
in_channels = 3
height = 2048  # Increased to 2048 to increase computation
width = 1280   # Increased to 1280 to increase computation
pool_size = (16, 16)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, pool_size]