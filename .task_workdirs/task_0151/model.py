import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, exponent):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.exponent = exponent

    def forward(self, x):
        x = F.conv_transpose1d(
            x,
            weight=self.conv_transpose1d.weight,
            bias=self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride,
            padding=self.conv_transpose1d.padding,
            output_padding=self.conv_transpose1d.output_padding,
        )
        x = torch.pow(x, self.exponent)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
length = 16384  # Increased from 512 to 16384 to increase computational load
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
exponent = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, exponent]