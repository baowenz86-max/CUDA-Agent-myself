import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, pooling_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.pooling_ratio = pooling_ratio
        self.maxpool3d = nn.MaxPool3d(kernel_size, stride=self.pooling_ratio)

    def forward(self, x):
        x = torch.sum(x, dim=1, keepdim=True)
        x = self.maxpool3d(x)
        return x


in_channels = 3
height, width, depth = 512, 256, 256  # Increased width and depth from 32 to 256
kernel_size = (2, 2, 2)
pooling_ratio = (2, 2, 2)


def get_inputs():
    return [torch.randn(4, in_channels, height, width, depth)]  # Increased batch size from 1 to 4


def get_init_inputs():
    return [in_channels, kernel_size, pooling_ratio]