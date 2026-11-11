import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias_shape):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )
        self.conv1d = nn.Conv1d(out_channels, out_channels, kernel_size, groups=groups)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose1d(x)
        x = F.conv1d(x, self.conv1d.weight, self.bias)
        return x


batch_size = 64
in_channels = 16
out_channels = 32
width = 8000  # Increased from 1000 to increase runtime
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
groups = 1
bias_shape = (out_channels,)


def get_inputs():
    return [torch.randn(batch_size, in_channels, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias_shape]