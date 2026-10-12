import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, bias_shape, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = x + self.bias
        x = self.bn(x)
        x = self.pool(x)
        x = torch.sigmoid(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 64, 64  # Increased from 32x32 to 64x64
kernel_size = 3
stride = 1
padding = 1
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, bias_shape, kernel_size, stride, padding]