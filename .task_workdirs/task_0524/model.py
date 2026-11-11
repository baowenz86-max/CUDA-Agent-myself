import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        maxpool_kernel_size,
        maxpool_stride,
        in_channels,
        out_channels,
        conv_kernel_size,
        conv_stride,
        padding,
        output_padding,
    ):
        super().__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            conv_kernel_size,
            stride=conv_stride,
            padding=padding,
            output_padding=output_padding,
        )

    def forward(self, x):
        x = self.maxpool(x)
        x = x.unsqueeze(2)  # Add an extra dimension to match the input shape for ConvTranspose2d
        x = self.conv_transpose(x)
        return x


batch_size = 128
maxpool_kernel_size = 3
maxpool_stride = 2
in_channels = 3
out_channels = 16
conv_kernel_size = 3
conv_stride = 2
padding = 1
output_padding = 1
sequence_length = 10000  # Increased to increase runtime


def get_inputs():
    return [torch.randn(batch_size, in_channels, sequence_length)]


def get_init_inputs():
    return [
        maxpool_kernel_size,
        maxpool_stride,
        in_channels,
        out_channels,
        conv_kernel_size,
        conv_stride,
        padding,
        output_padding,
    ]