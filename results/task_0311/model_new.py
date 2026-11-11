import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        return x

    def debug_forward(self, x):
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        yield 'step_001', x
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        yield 'step_002', x
        yield 'step_003', x
