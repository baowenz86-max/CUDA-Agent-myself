import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, exponent):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        self.pow = torch.nn.Parameter(torch.tensor(exponent))

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.pow(x, self.pow)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 1024, 1024  # Increased spatial dimensions to increase computation
kernel_size = 3
stride = 2
padding = 1
exponent = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, exponent]