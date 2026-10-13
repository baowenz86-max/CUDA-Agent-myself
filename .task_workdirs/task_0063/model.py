import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, output_size, in_channels, kernel_size, stride, padding):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=output_size)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.unfold(x)
        return x


batch_size = 64
in_channels = 128
length = 512  # Changed from 256 to 512 to ensure at least one dimension >=512
output_size = 4050  # Increased to increase runtime
kernel_size = 3
stride = 1
padding = 0


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [output_size, in_channels, kernel_size, stride, padding]