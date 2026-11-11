import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pooling_size):
        super().__init__()
        self.pooling_size = pooling_size

    def forward(self, x):
        x = cuda_extension.clamp_forward(x, 0.0, 1.0)
        x = cuda_extension.adaptive_avg_pool2d_forward(x, self.pooling_size)
        return x

    def debug_forward(self, x):
        x = cuda_extension.clamp_forward(x, 0.0, 1.0)
        yield 'step_001', x
        x = cuda_extension.adaptive_avg_pool2d_forward(x, self.pooling_size)
        yield 'step_002', x
        yield 'step_003', x
