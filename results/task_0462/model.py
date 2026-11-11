import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.avg_pool3d = nn.AvgPool3d(kernel_size)

    def forward(self, x):
        x = self.avg_pool3d(x)
        return x


batch_size = 16
in_channels = 3
kernel_size = (2, 3, 3)
depth = 512
height = 128  # Increased from 32 to 128
width = 128   # Increased from 32 to 128


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, kernel_size]