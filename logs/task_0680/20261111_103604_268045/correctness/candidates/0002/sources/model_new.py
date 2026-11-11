import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        x = cuda_extension.custom_cat(x1, x2, self.dim)
        x = cuda_extension.custom_nansum(x, self.dim)
        return x

    def debug_forward(self, x1, x2):
        x = cuda_extension.custom_cat(x1, x2, self.dim)
        yield 'step_001', x
        x = cuda_extension.custom_nansum(x, self.dim)
        yield 'step_002', x
        yield 'step_003', x
