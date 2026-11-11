import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = cuda_extension.abs_forward(x)
        x = cuda_extension.softplus_forward(x)
        return x

    def debug_forward(self, x):
        x_abs = cuda_extension.abs_forward(x)
        yield 'step_001', x_abs
        x_softplus = cuda_extension.softplus_forward(x_abs)
        yield 'step_002', x_softplus
        yield 'step_003', x_softplus
