import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, kernel_size, dilation, padding):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding)

    def forward(self, x):
        x = self.unfold(x)
        x = torch.sign(x)
        return x


batch_size = 16
in_channels = 3
height = 1024  # Increased from 512 to increase computation
width = 512    # Increased from 256 to increase computation
kernel_size = (3, 5)
dilation = (2, 1)
padding = (1, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, dilation, padding]