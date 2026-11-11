import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias, self.linear.in_features, self.linear.out_features)
        x = cuda_extension.minimum_forward(x, self.bias)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias, self.linear.in_features, self.linear.out_features)
        yield 'step_001', x
        x = cuda_extension.minimum_forward(x, self.bias)
        yield 'step_002', x
        yield 'step_003', x
