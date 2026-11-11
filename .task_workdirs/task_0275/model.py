import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(num_features)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.gelu(x)
        return x


def get_inputs():
    batch_size = 16
    num_channels = 3
    height, width = 1024, 512  # Increased spatial dimensions to increase runtime
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    return [3]  # num_features matches num_channels