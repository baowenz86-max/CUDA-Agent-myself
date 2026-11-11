import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = x + self.bias
        x, _ = torch.max(x, dim=1)  # torch.max operation
        x = torch.sqrt(x)  # torch.sqrt operation
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
in_channels = 3
out_channels = 16
height, width = 256, 256  # Increased input spatial dimensions to increase runtime
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]