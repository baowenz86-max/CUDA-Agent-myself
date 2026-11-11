import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.bias
        x = torch.softmax(x, dim=1)
        x = x * self.scaling_factor
        return x


batch_size = 512
in_channels = 3
out_channels = 16
length = 8192  # Increased from 32 to 8192 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]