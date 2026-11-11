import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, output_size, pool_size):
        super().__init__()
        self.stack = nn.AdaptiveAvgPool1d(output_size)
        self.pool_size = pool_size

    def forward(self, x):
        x = self.stack(x)
        x = torch.stack([x] * self.pool_size, dim=1)
        return x


batch_size = 64
input_size = 1024
output_size = 64
pool_size = 16


def get_inputs():
    return [torch.randn(batch_size, input_size, 4096)]


def get_init_inputs():
    return [input_size, output_size, pool_size]