import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = cuda_extension.hardsigmoid_forward(x)
        y = cuda_extension.relu_forward(y)
        return x, y

    def debug_forward(self, x, y):
        x = cuda_extension.hardsigmoid_forward(x)
        yield 'step_001', x
        y = cuda_extension.relu_forward(y)
        yield 'step_002', y
        yield 'step_003', (x, y)
