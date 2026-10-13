import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, negative_slope=0.01)  # Apply leaky ReLU
        x = x + self.bias
        return x


batch_size = 512  # Increased to 512 to ensure at least one dimension >=512
in_channels = 3
out_channels = 16
height, width = 128, 128  # Increased spatial dimensions to increase computation
kernel_size = 3
bias_shape = (out_channels, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]