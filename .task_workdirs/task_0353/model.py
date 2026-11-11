import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = F.relu(x)
        x = self.conv(x)
        return x


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
stride = 1
padding = 1
bias = False
input_height = 512  # Changed from 256 to 512
input_width = 512  # Changed from 256 to 512


def get_inputs():
    x = torch.randn(batch_size, in_channels, input_height, input_width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias]