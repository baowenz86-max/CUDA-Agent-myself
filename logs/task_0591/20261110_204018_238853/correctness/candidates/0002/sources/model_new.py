import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = cuda_extension.gelu_forward(x)
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.gelu_forward(x)
        yield 'step_001', x
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        yield 'step_002', x
        yield 'step_003', x
