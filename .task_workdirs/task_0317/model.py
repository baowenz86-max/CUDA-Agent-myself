import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.conv_transpose = nn.LazyConvTranspose3d(
            out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x):
        return self.conv_transpose(x)


batch_size = 64  # Reduced from 512 to lower runtime
in_channels = 16
out_channels = 32
depth, height, width = 24, 32, 32
kernel_size = (3, 4, 5)
stride = (1, 2, 2)
padding = (1, 2, 3)
output_padding = (0, 0, 0)
bias = False


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias]