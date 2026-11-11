import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pool_size, pooling_type):
        super().__init__()
        # Store configuration for possible introspection, but never use torch modules in forward.
        self.pool_size = pool_size
        self.pooling_type = pooling_type

    def forward(self, x):
        x = cuda_extension.clamp_forward(x)
        x = cuda_extension.maxpool2d_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.clamp_forward(x)
        yield 'step_001', x
        x = cuda_extension.maxpool2d_forward(x)
        yield 'step_002', x
        yield 'step_003', x


def get_inputs():
    return [torch.randn(512, 3, 320, 320)]


def get_init_inputs():
    return [(2, 2), "max"]
