import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, kernel_size, dilation, height, width):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        x = self.unfold(x)
        return x


batch_size = 48  # Increased from 16 to 48 to increase runtime
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
dilation = (1, 1)
height = 512
width = 256


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [batch_size, in_channels, out_channels, kernel_size, dilation, height, width]