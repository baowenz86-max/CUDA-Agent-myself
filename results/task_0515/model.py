import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, ceil_mode):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(kernel_size)
        self.linear = nn.Linear(in_channels * kernel_size[0] * kernel_size[1], out_channels)

    def forward(self, x):
        x = self.adaptive_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.linear(x)
        return x


# Parameters for the model
in_channels = 3
out_channels = 16
height, width = 512, 512  # Increased width from 64 to 512
kernel_size = (2, 2)
stride = 1
padding = 0
ceil_mode = False


def get_inputs():
    return [torch.randn(4, in_channels, height, width)]  # Increased batch size from 1 to 4


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, ceil_mode]