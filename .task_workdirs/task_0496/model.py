import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)  # Assuming 1-dimensional input for simplicity
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)

    def forward(self, x):
        x = torch.clamp(x, min=-1.0, max=1.0)  # Clamping the input tensor to a range [-1, 1]
        x = self.conv(x)
        x = self.pool(x)
        return x


batch_size = 128
in_channels = 1
out_channels = 64
kernel_size = 3
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0
seq_len = 10240  # Increased sequence length to increase computation time


def get_inputs():
    return [torch.randn(batch_size, in_channels, seq_len)]  # Use updated seq_len


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size, pool_stride, pool_padding]