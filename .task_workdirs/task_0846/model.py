import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x, y):
        x = F.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride,
            padding=self.conv_transpose1d.padding,
            output_padding=self.conv_transpose1d.output_padding,
        )
        return torch.maximum(x, y)


batch_size = 128
in_channels = 3
out_channels = 16
width = 10240  # Increased from 512 to 10240 to increase runtime
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias = True


def get_inputs():
    x = torch.randn(batch_size, in_channels, width)
    y = torch.randn(batch_size, out_channels, width * stride)
    return [x, y]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias]