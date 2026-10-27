import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, pool_size):
        super().__init__()
        self.gelu = nn.GELU()
        self.max_pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        x = self.gelu(x)
        x = self.max_pool(x)
        return x


batch_size = 16
in_channels = 64
pool_size = 32
sequence_length = 102400  # Increased from 1024 to 102400 to increase runtime


def get_inputs():
    x = torch.randn(batch_size, in_channels, sequence_length)
    return [x]


def get_init_inputs():
    return [in_channels, pool_size]