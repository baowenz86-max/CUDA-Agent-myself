import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.pool(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 64
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, 50000)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]