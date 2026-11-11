import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, pool_size):
        super().__init__()
        self.in_channels = in_channels
        self.pool_size = pool_size

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.max_pool1d_forward(x, self.pool_size)
        return x

    def debug_forward(self, x):
        relu_out = cuda_extension.relu_forward(x)
        yield 'step_001', relu_out
        max_pool_out = cuda_extension.max_pool1d_forward(relu_out, self.pool_size)
        yield 'step_002', max_pool_out
        yield 'step_003', max_pool_out
