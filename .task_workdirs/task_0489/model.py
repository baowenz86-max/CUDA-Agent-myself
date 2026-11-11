import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.adaptive_avg_pool(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + self.bias
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 512, 512  # Increased from 32x32 to 512x512
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]