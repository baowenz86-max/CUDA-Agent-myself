import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        another_in_channels,
        conv1d_weight_shape,
        conv1d_bias_shape,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.conv1d_weight = nn.Parameter(torch.randn(conv1d_weight_shape))
        self.conv1d_bias = nn.Parameter(torch.randn(conv1d_bias_shape)) if conv1d_bias_shape is not None else None
        self.another_in_channels = another_in_channels

    def forward(self, x, y):
        x = self.conv_transpose(x)
        x = torch.add(x, self.bias)
        y = F.conv1d(y, self.conv1d_weight, self.conv1d_bias, stride=1, padding=0)
        return x, y


batch_size = 512
in_channels = 64  # Increased from 32
out_channels = 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 0
bias_shape = (out_channels, 1)
another_in_channels = 128  # Increased from 64
conv1d_weight_shape = (128, another_in_channels, 3)
conv1d_bias_shape = 128


def get_inputs():
    x = torch.randn(batch_size, in_channels, 1024)  # Increased sequence length from 16
    y = torch.randn(batch_size, another_in_channels, 2048)  # Increased sequence length from 14
    return [x, y]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        another_in_channels,
        conv1d_weight_shape,
        conv1d_bias_shape,
    ]