import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, values_shape):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.values = nn.Parameter(torch.randn(values_shape))

    def forward(self, x):
        mask = torch.isin(x, self.values)
        x = self.maxpool(x)
        return x


batch_size = 512
in_channels = 3
height, width = 128, 128  # Increased from 64x64 to 128x128 to increase runtime
kernel_size = 3
stride = 2
padding = 1
values_shape = (10,)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, values_shape]