import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        mean_x = cuda_extension.mean_dim(x, self.dim, True)
        return cuda_extension.cat_dim(mean_x, x, self.dim)

    def debug_forward(self, x):
        mean_x = cuda_extension.mean_dim(x, self.dim, True)
        yield 'step_001', mean_x
        out = cuda_extension.cat_dim(mean_x, x, self.dim)
        yield 'step_002', out
