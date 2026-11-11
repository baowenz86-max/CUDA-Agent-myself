import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.adaptive_avg_pool(x)
        x = self.conv(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 256, 256  # Increased spatial dimensions to increase computation
kernel_size = (2, 2)
stride = (2, 2)
padding = (0, 0)
bias = False


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias]