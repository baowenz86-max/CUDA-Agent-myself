import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.fused_relu_hardsigmoid(x)
        return x

    def debug_forward(self, x):
        y_relu = cuda_extension.fused_relu_hardsigmoid(x)
        yield 'step_001', y_relu
        yield 'step_002', y_relu
        yield 'step_003', y_relu
