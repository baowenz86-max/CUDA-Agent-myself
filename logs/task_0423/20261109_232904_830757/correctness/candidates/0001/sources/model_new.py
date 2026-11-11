import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, bias_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.custom_clamp_min_max(x)
        x = cuda_extension.custom_triu(x)
        x = cuda_extension.custom_add_bias(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_clamp_min_max(x)
        yield 'step_001', x
        x = cuda_extension.custom_triu(x)
        yield 'step_002', x
        x = cuda_extension.custom_add_bias(x, self.bias)
        yield 'step_003', x
        yield 'step_004', x
