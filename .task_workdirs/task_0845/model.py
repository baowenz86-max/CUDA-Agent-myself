import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(in_channels)
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias_shape,
        )

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.conv_transpose(x)
        return x


batch_size = 128
in_channels = 16
out_channels = 32
length = 8192  # Increased from 512 to 8192 to increase computational load
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = False


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]