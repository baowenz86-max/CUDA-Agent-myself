import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.batchnorm2d(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x


batch_size = 16
in_channels = 3
out_channels = 64
height = 512
width = 512  # Increased from 256 to 512 to increase computation
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]