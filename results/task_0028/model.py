import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, kernel_size, dilation, padding, unfolded_channels):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size, dilation=dilation, padding=padding)
        self.unfolded_channels = unfolded_channels

    def forward(self, x):
        x = self.unfold(x)
        x = torch.sign(x)
        x = torch.cos(x)
        return x


batch_size = 18  # Increased from 16 to 18 to increase runtime
in_channels = 3
height = 512
width = 256
kernel_size = (3, 5)
dilation = (2, 1)
padding = (1, 2)


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, dilation, padding, in_channels * kernel_size[0] * kernel_size[1]]