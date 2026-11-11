import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = torch.mean(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 512, 512  # Increased width to 512 to increase computational load
kernel_size = 3


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]