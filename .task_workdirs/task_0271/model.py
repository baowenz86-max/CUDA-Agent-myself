import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(output_size)

    def forward(self, x):
        return self.adaptive_avg_pool3d(x)


batch_size = 2
in_channels = 64
depth = 512
height = 128  # Increased from 10 to 128 to increase computation
width = 128   # Increased from 10 to 128 to increase computation
output_size = (1, 1, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [output_size]