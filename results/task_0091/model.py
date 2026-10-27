import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = torch.triu(x)
        x = self.conv(x)
        x = x + self.bias
        return x


batch_size = 512
in_channels = 3
out_channels = 64
height, width = 64, 64  # Increased from 32x32 to 64x64
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]