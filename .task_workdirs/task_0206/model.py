import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = F.leaky_relu(x, 0.01)  # LeakyReLU with negative slope of 0.01
        x = torch.abs(x)  # absolute value
        return x


batch_size = 16
in_channels = 1024  # Increased from 3 to increase input size
out_channels = 64
kernel_size = 3
stride = 2
padding = 1
height = 512
width = 256


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]