import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_size, num_params):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pad = nn.ReflectionPad2d(padding_size)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(out_channels)) for _ in range(num_params)])

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
in_channels = 3
out_channels = 16
kernel_size = 3
padding_size = 1
height, width = 128, 128  # Increased spatial dimensions to increase runtime
num_params = 5


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding_size, num_params]