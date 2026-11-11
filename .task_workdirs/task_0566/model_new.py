import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = cuda_extension.abs_forward(x)
        x = cuda_extension.gt_scalar_forward(x, 0.5)
        return x

    def debug_forward(self, x):
        x = cuda_extension.abs_forward(x)
        yield 'step_001', x
        y = cuda_extension.gt_scalar_forward(x, 0.5)
        yield 'step_002', y
        yield 'step_003', y
