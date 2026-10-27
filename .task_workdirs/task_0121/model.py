import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.padding = nn.ConstantPad1d(padding, 0)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.padding(x)
        x = self.conv_transpose(x)
        x = x + self.bias
        return x


batch_size = 512
in_channels = 3
out_channels = 64
kernel_size = 3
stride = 2
padding = (2, 3)
output_padding = 1
bias_shape = (out_channels, 1, 1)


def get_inputs():
    x = torch.randn(batch_size, in_channels, 32, 32)  # Increased spatial dimensions from 8x8 to 32x32
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]