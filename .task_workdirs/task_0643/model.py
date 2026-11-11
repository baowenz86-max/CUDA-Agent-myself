import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, weight_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.weight = nn.Parameter(torch.randn(weight_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.einsum("bct,cd->bdt", x, self.weight)
        return x


batch_size = 256  # Increased from 64 to 256 to increase computation
in_channels = 3
out_channels = 16
length = 4096  # Increased from 512 to 4096 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
weight_shape = (out_channels, 32)


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, weight_shape]