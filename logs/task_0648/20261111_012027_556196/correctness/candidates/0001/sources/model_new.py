import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.mean_reduce_op(x, dim=(1, 2, 3), keepdim=True)
        x = cuda_extension.remainder_op(x, 2.0)
        return x

    def debug_forward(self, x):
        x = cuda_extension.mean_reduce_op(x, dim=(1, 2, 3), keepdim=True)
        yield 'step_001', x
        x = cuda_extension.remainder_op(x, 2.0)
        yield 'step_002', x
        yield 'step_003', x
