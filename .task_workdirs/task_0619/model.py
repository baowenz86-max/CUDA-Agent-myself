import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = self.pool(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 128, 128  # Increased from 32x32 to 128x128


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    kernel_size = (3, 3)
    pool_kernel_size = (2, 2)
    pool_stride = (2, 2)
    pool_padding = (0, 0)
    return [3, 16, kernel_size, pool_kernel_size, pool_stride, pool_padding]