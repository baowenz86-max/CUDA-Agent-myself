import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.custom_transpose(x, self.dim)
        x = cuda_extension.custom_cat_2x(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_transpose(x, self.dim)
        yield 'step_001', x
        x = cuda_extension.custom_cat_2x(x)
        yield 'step_002', x
        yield 'step_003', x
