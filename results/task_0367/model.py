import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pooling_kernel):
        super().__init__()
        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d(pooling_kernel)

    def forward(self, x):
        x = self.adaptive_avg_pool_2d(x)
        return x


batch_size = 16
in_channels = 3
height = 1024  # Increased from 512 to increase computation
width = 1024   # Increased from 256 to increase computation
pooling_kernel = (5, 5)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [pooling_kernel]