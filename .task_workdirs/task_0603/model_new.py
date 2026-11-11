import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        x = cuda_extension.custom_gelu(x)
        x = cuda_extension.custom_mean_dim1(x)
        return x

    def debug_forward(self, x):
        y = cuda_extension.custom_gelu(x)
        yield 'step_001', y
        z = cuda_extension.custom_mean_dim1(y)
        yield 'step_002', z
        yield 'step_003', z
