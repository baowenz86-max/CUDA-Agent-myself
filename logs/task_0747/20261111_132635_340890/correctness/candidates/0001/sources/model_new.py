import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, reduction_dim, transpose_dim):
        super().__init__()
        self.reduction_dim = reduction_dim
        self.transpose_dim = transpose_dim

    def forward(self, x):
        x = cuda_extension.mean_only(x, self.reduction_dim)
        x = cuda_extension.transpose_only(x, self.transpose_dim[0], self.transpose_dim[1])
        return x

    def debug_forward(self, x):
        x = cuda_extension.mean_only(x, self.reduction_dim)
        yield 'step_001', x
        x = cuda_extension.transpose_only(x, self.transpose_dim[0], self.transpose_dim[1])
        yield 'step_002', x
        yield 'step_003', x
