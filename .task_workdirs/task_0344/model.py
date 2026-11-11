import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding, input_channels, output_channels, kernel_size, bias):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, bias=bias))

    def forward(self, x, indices, dim):
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = torch.gather(x, dim, indices)
        return x


batch_size = 512
input_channels = 3
output_channels = 16
height, width = 128, 128  # Increased from 32x32 to 128x128
kernel_size = 3
padding = 2
dim = 1
bias = True


def get_inputs():
    x = torch.randn(batch_size, input_channels, height, width)
    indices = torch.randint(0, output_channels, (batch_size, input_channels, height, width))
    return [x, indices, dim]


def get_init_inputs():
    return [padding, input_channels, output_channels, kernel_size, bias]