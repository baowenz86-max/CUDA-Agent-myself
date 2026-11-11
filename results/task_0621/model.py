import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.abs = nn.Sequential(nn.AvgPool2d(kernel_size, stride=stride, padding=padding))

    def forward(self, x):
        x = torch.abs(x)
        x = self.abs(x)
        return x


batch_size = 68
in_channels = 3
out_channels = 64
height = 512
width = 256
kernel_size = (3, 5)
stride = 1
padding = (1, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]