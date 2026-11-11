import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        x = cuda_extension.mean_dim_last(x)
        x = cuda_extension.softplus(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.mean_dim_last(x)
        yield 'step_001', x
        x = cuda_extension.softplus(x)
        yield 'step_002', x
        yield 'step_003', x
