import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = F.conv_transpose1d
        self.bitwise_xor = torch.bitwise_xor
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.bias
        # x = self.bitwise_xor(x)  # torch.bitwise_xor is not implemented for 2D tensors
        return x


batch_size = 128
in_channels = 3
out_channels = 16
width = 10240  # Increased from 512 to 10240 to increase computational load
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]