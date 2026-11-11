import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = F.conv_transpose1d(
            x, self.conv_transpose.weight, self.bias, self.conv_transpose.stride, self.conv_transpose.padding
        )
        return torch.addcmul(x, x, x)


batch_size = 128
in_channels = 3
out_channels = 16
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels,)
seq_len = 16384  # Increased sequence length to increase runtime


def get_inputs():
    return [torch.randn(batch_size, in_channels, seq_len)]  # Use updated seq_len


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]