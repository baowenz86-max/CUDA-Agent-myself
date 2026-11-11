import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, normalized_shape, eps):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        x = cuda_extension.abs_forward(x)
        x = cuda_extension.layer_norm_forward(x, len(self.normalized_shape), self.eps)
        return x

    def debug_forward(self, x):
        x_abs = cuda_extension.abs_forward(x)
        yield 'step_001', x_abs
        x_ln = cuda_extension.layer_norm_forward(x_abs, len(self.normalized_shape), self.eps)
        yield 'step_002', x_ln
        yield 'step_003', x_ln
