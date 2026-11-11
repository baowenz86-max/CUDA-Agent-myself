import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        # Conv3d layer
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
        # Bias for the conv3d
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Conv3d followed by GELU activation, then addition of bias
        x = self.conv3d(x)
        x = F.gelu(x)
        x = x + self.bias
        return x


# Initialization parameters
in_channels = 3
out_channels = 16
kernel_size = (3, 3, 3)
stride = (1, 1, 1)
padding = (1, 1, 1)
bias_shape = (out_channels, 1, 1, 1)
# Adjust spatial dimensions to increase computation
depth = 512
height = 64  # Increased from 20
width = 64   # Increased from 20


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]


def get_inputs():
    return [torch.randn(1, in_channels, depth, height, width)]  # Larger spatial dimensions