import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x):
        x = torch.logical_or(x > 0.5, x < -0.5)  # Logical OR operation is applied to the condition x > 0.5 and x < -0.5
        return self.conv2d(x.float())


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
width = 512  # Adjusted from 256 to 512 to ensure at least one dimension >=512
height = 512  # Increased from 256 to 512 to increase runtime
stride = 1
padding = (1, 2)
dilation = (1, 1)
bias = False


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, bias]