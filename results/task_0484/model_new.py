import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.custom_mean_dim(x, self.dim)
        x = cuda_extension.custom_softmax_lastdim(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_mean_dim(x, self.dim)
        yield 'step_001', x
        x = cuda_extension.custom_softmax_lastdim(x)
        yield 'step_002', x
        yield 'step_003', x
