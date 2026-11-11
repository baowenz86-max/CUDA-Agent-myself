import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, search_values, max_pool_kernel_size):
        super().__init__()
        self.isin_values = nn.Parameter(torch.tensor(search_values), requires_grad=False)
        self.max_pool = nn.MaxPool1d(max_pool_kernel_size)

    def forward(self, x):
        mask = torch.isin(x, self.isin_values)
        x = self.max_pool(x)
        return x, mask


search_values = [0.5, 0.7, 1.0]
max_pool_kernel_size = 3
batch_size = 16
input_channels = 128
length = 16384  # Increased from 1024 to 16384 to increase runtime


def get_inputs():
    x = torch.randn(batch_size, input_channels, length)
    return [x]


def get_init_inputs():
    return [search_values, max_pool_kernel_size]