import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels, pool_size):
        super().__init__()
        self.stack = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        stack = [self.max_pool(x)]
        return torch.stack(stack)


batch_size = 512  # Increased from 16 to 512 to increase runtime
input_channels = 3
height = 512
width = 256
pool_size = 2


def get_inputs():
    x = torch.randn(batch_size, input_channels, height, width)
    return [x]


def get_init_inputs():
    return [input_channels, pool_size]