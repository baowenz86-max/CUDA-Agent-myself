import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_size, reduction_factor, bias):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        self.pooling = nn.AdaptiveAvgPool1d(pooling_size)
        self.reduction_factor = reduction_factor

    def forward(self, x):
        x = self.conv1d(x)
        x = self.pooling(x)
        x = x[:, :, : self.reduction_factor]
        x = torch.std(x, dim=2)
        return x


batch_size = 512
in_channels = 8
out_channels = 16
kernel_size = 5
pooling_size = 200  # Increased from 10
reduction_factor = 50  # Increased from 5
bias = True


def get_inputs():
    x = torch.randn(batch_size, in_channels, pooling_size * reduction_factor)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pooling_size, reduction_factor, bias]