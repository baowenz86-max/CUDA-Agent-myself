import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.sum_dim(x, 1, True)
        x = cuda_extension.leaky_relu(x, 0.01)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.sum_dim(x, 1, True)
        yield 'step_001', x1
        x2 = cuda_extension.leaky_relu(x1, 0.01)
        yield 'step_002', x2
        yield 'step_003', x2
