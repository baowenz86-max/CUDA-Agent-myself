import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = torch.abs(x)
        x = self.max_pool(x)
        return x


batch_size = 16
in_channels = 3
out_channels = 64
height = 2048  # Increased from 512
width = 1024   # Increased from 256
kernel_size = (3, 5)
stride = 1
padding = (1, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]