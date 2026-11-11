import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        x = F.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            bias=self.conv_transpose1d.bias if self.bias else None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        x = self.silu(x)
        return x


batch_size = 128
in_channels = 8
out_channels = 16
length = 10240  # Increased from 512 to 10240 to increase runtime
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias = True


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias]