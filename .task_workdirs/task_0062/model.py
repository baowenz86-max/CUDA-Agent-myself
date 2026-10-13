import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
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

    def forward(self, x):
        x = self.conv_transpose1d(x)
        x = F.sigmoid(x)
        return x


batch_size = 64
in_channels = 16
out_channels = 32
kernel_size = 4
stride = 2
padding = 1
output_padding = 0
seq_len = 16384  # Increased sequence length to increase computation


def get_inputs():
    return [torch.randn(batch_size, in_channels, seq_len)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]