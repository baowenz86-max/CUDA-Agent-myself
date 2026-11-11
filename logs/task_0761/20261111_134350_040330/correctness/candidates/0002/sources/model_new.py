import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim, epsilon=1e-8):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

    def forward(self, x):
        x = cuda_extension.softmax1d_forward(x, self.dim)
        x = cuda_extension.squeeze_forward(x)
        x = cuda_extension.add_scalar_forward(x, self.epsilon)
        return x

    def debug_forward(self, x):
        x = cuda_extension.softmax1d_forward(x, self.dim)
        yield 'step_001', x
        x = cuda_extension.squeeze_forward(x)
        yield 'step_002', x
        x = cuda_extension.add_scalar_forward(x, self.epsilon)
        yield 'step_003', x
        yield 'step_004', x
