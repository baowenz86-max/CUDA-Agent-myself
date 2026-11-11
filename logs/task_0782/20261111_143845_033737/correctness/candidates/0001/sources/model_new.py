import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x, y):
        x = cuda_extension.mul_forward(x, y)
        return x

    def debug_forward(self, x, y):
        z = cuda_extension.mul_forward(x, y)
        yield ('step_001', z)
        yield ('step_002', z)
