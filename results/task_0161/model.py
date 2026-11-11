import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, pool_size):
        super().__init__()
        self.avg_pool_3d = nn.AvgPool3d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.avg_pool_3d(x)
        x = torch.sum(x, dim=1)  # sum over the channel dimension
        return x


def get_inputs():
    batch_size, in_channels, depth, height, width = 512, 1, 32, 128, 128
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [1, (2, 2, 2)]