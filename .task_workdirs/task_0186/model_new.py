import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = cuda_extension.reduce_mean_dim1(x)
        x = cuda_extension.mul_scalar(x, float(self.out_channels))
        return x

    def debug_forward(self, x):
        x = cuda_extension.reduce_mean_dim1(x)
        yield 'step_001', x
        x = cuda_extension.mul_scalar(x, float(self.out_channels))
        yield 'step_002', x
        yield 'step_003', x
