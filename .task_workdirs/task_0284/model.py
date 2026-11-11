import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride=stride)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.glu(x)
        x = torch.min(x, dim=1)[0]
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 128, 128  # Increased from 32x32 to 128x128
kernel_size = 3
stride = 2
padding = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]