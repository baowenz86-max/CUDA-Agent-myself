import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_size):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding_size)

    def forward(self, x):
        x = self.pad(x)
        x = torch.ge(x, 0)  # torch.ge performs element-wise greater-than-equal operation
        x = torch.remainder(x, 2)  # torch.remainder performs element-wise remainder operation
        return x


batch_size = 16
in_channels = 32  # Increased from 1 to 32 to increase computation
height, width = 512, 256
kernel_size = 3
padding_size = (2, 3, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, kernel_size, padding_size]