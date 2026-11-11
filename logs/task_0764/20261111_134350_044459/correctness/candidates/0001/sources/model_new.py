import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.sum_over_dim0(x, self.dim)
        x = cuda_extension.softmax_over_dim0(x, self.dim)
        return x

    def debug_forward(self, x):
        summed = cuda_extension.sum_over_dim0(x, self.dim)
        yield 'step_001', summed
        softmaxed = cuda_extension.softmax_over_dim0(summed, self.dim)
        yield 'step_002', softmaxed
        yield 'step_003', softmaxed
