import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x):
        x = self.conv3d(x)
        x = torch.sum(x, dim=1)  # Sum along the channel dimension
        return x


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 5)
depth = 32
height = 256
width = 256  # Reduced width from 512 to 256 to decrease runtime
stride = 1
padding = (1, 2, 2)
dilation = (2, 1, 1)
bias = True


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, bias]