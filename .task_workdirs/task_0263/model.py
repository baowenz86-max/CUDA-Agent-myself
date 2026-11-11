import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.avg_pool2d = nn.AvgPool2d(kernel_size)
        self.max_pool2d = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        x_avg = self.avg_pool2d(x)
        x_max = self.max_pool2d(x)
        return x_avg, x_max


batch_size = 16
in_channels = 3
height, width = 2048, 2048  # Increased spatial dimensions to increase computation
kernel_size = (2, 2)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, kernel_size]