import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.leaky_relu_forward(x, negative_slope=0.01)
        return x

    def debug_forward(self, x):
        x = cuda_extension.leaky_relu_forward(x, negative_slope=0.01)
        yield 'step_001', x
        yield 'step_002', x
