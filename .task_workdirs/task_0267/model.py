import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.conv_transpose = nn.LazyConvTranspose2d(
            out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x):
        return self.conv_transpose(x)


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 64, 64  # Increased from 32x32 to 64x64
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias = False


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias]