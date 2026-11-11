import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(maxpool_kernel_size, stride=maxpool_stride, padding=maxpool_padding)

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = self.conv(x)
        x = self.maxpool(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 512, 64  # Increased width from 32 to 64
kernel_size = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
maxpool_padding = 0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
    ]