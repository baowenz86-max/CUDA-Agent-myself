import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(in_channels)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.batchnorm2d(x)
        x = self.hardswish(x)
        return x


batch_size = 512
in_channels = 64
out_channels = 64
height, width = 64, 64  # Increased from 32x32 to 64x64
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]