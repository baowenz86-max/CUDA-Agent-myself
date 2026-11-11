import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, flip_dims):
        super().__init__()
        self.flip_dims = flip_dims

    def forward(self, x):
        x = cuda_extension.custom_abs_forward(x)
        x = cuda_extension.custom_flip_forward(x, self.flip_dims[0], self.flip_dims[1])
        return x

    def debug_forward(self, x):
        x_abs = cuda_extension.custom_abs_forward(x)
        yield 'step_001', x_abs
        x_flip = cuda_extension.custom_flip_forward(x_abs, self.flip_dims[0], self.flip_dims[1])
        yield 'step_002', x_flip
        yield 'step_003', x_flip
