import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.selu_forward(x)
        return x

    def debug_forward(self, x):
        y = cuda_extension.relu_forward(x)
        yield 'step_001', y
        z = cuda_extension.selu_forward(y)
        yield 'step_002', z
        yield 'step_003', z
