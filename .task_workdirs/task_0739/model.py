import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, pool_size):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=pool_size)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = F.softmax(x, dim=1)
        return x


batch_size = 16
in_channels = 3
height = 5120  # Increased from 512
width = 2560   # Increased from 256
pool_size = (16, 16)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, pool_size]