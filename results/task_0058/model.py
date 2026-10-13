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
        conv1d_out_channels,
        conv1d_kernel_size,
        conv1d_stride,
        conv1d_padding,
        dim,
    ):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.conv1d = nn.Conv1d(
            out_channels, conv1d_out_channels, conv1d_kernel_size, stride=conv1d_stride, padding=conv1d_padding
        )
        self.dim = dim

    def forward(self, x):
        x = F.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            bias=self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride,
            padding=self.conv_transpose1d.padding,
            output_padding=self.conv_transpose1d.output_padding,
        )
        x = self.conv1d(x)
        x = torch.sum(x, dim=self.dim)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
length = 8192  # Increased from 512 to 8192 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
conv1d_out_channels = 8
conv1d_kernel_size = 3
conv1d_stride = 1
conv1d_padding = 1
dim = 1  # Sum along the length dimension


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        conv1d_out_channels,
        conv1d_kernel_size,
        conv1d_stride,
        conv1d_padding,
        dim,
    ]