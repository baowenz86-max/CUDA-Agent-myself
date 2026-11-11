import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        x = self.maxpool(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        return x


batch_size = 64
in_channels = 3
out_channels = 16
height, width = 1024, 1024  # Increased spatial dimensions to increase runtime
kernel_size = (2, 2)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]