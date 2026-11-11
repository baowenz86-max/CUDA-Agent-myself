import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, kernel_size1, kernel_size2, stride, padding):
        super().__init__()
        self.pool1 = nn.AvgPool3d(kernel_size=kernel_size1, stride=stride, padding=padding)
        self.pool2 = nn.AvgPool3d(kernel_size=kernel_size2, stride=stride, padding=padding)

    def forward(self, x):
        x = self.pool1(x)
        x = self.pool2(x)
        return x


batch_size = 16
in_channels = 3
depth, height, width = 512, 128, 128  # Increased height and width from 32 to 128
kernel_size1 = (1, 3, 3)
kernel_size2 = (1, 2, 2)
stride = 2
padding = (0, 1, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size1, kernel_size2, stride, padding]