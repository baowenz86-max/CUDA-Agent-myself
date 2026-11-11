import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.linear_forward(x, self.linear.weight, self.bias)
        return x

    def debug_forward(self, x):
        out = cuda_extension.linear_forward(x, self.linear.weight, self.bias)
        yield 'step_001', out
        yield 'step_002', out
