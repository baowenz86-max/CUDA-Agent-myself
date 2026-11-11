import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, num_units, kernel_size, stride):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.num_units = num_units

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.avg_pool(x)
        return x


batch_size = 512
in_channels = 3
num_units = 16
height, width = 128, 128  # Increased from 32x32 to 128x128
kernel_size = 3
stride = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, num_units, kernel_size, stride]