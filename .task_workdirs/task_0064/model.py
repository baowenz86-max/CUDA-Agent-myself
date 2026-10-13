import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels, padding_size, out_channels):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding_size)
        self.conv = nn.Conv2d(input_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = torch.nansum(x, dim=[1, 2])  # sum across the height and width dimensions
        return x


batch_size = 512
input_channels = 3
padding_size = (1, 1, 1, 1)  # padding for each dimension (left, right, top, bottom)
out_channels = 16
height, width = 128, 128  # Increased from 32x32 to 128x128 to increase computation


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [input_channels, padding_size, out_channels]