import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.max_pool3d = nn.MaxPool3d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.max_pool3d(x)


batch_size = 1024
in_channels = 3
depth, height, width = 32, 32, 32
kernel_size = (1, 2, 2)
stride = (1, 2, 2)
padding = (0, 1, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]