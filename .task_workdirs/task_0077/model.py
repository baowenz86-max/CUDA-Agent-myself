import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(bias_shape)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.norm(x)
        return x


batch_size = 1024
in_channels = 3
out_channels = 16
length = 4096  # Increased from 32 to increase computation
kernel_size = 3
padding = 1
bias_shape = (out_channels, length)


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding, bias_shape]