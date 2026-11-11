import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_groups, num_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.conv_transpose1d = nn.ConvTranspose1d(
            num_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        x = F.group_norm(x, self.num_groups)
        x = self.conv_transpose1d(x)
        return x


batch_size = 128
num_groups = 4
num_channels = 16
out_channels = 32
height = 8192  # Increased from 512 to 8192 to increase runtime
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias = True


def get_inputs():
    return [torch.randn(batch_size, num_channels, height)]


def get_init_inputs():
    return [num_groups, num_channels, out_channels, kernel_size, stride, padding, output_padding, bias]