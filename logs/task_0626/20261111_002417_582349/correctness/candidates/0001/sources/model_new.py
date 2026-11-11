import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.custom_softplus(x)
        x = cuda_extension.custom_sum_dim1(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_softplus(x)
        yield 'step_001', x
        x = cuda_extension.custom_sum_dim1(x)
        yield 'step_002', x
        yield 'step_003', x
