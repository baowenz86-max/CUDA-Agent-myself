import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(in_channels // 2)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = torch.erf(x)
        x = torch.sin(x)
        return x


batch_size = 16
in_channels = 32  # Increased from 3 to increase computation
out_channels = 64
height = 512
width = 256


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels]