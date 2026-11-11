import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.detach(x)
        return x


batch_size = 32
in_channels = 16
out_channels = 32
length = 32768  # Increased from 512 to 32768 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias = True


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias]