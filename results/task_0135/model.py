import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        x = torch.trunc(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 512, 1024  # Increased width to increase computation
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]