import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = cuda_extension.custom_linear(x, self.linear.weight, self.linear.bias)
        x = cuda_extension.custom_gelu(x)
        x = cuda_extension.custom_softmax(x, 1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_linear(x, self.linear.weight, self.linear.bias)
        yield 'step_001', x
        x = cuda_extension.custom_gelu(x)
        yield 'step_002', x
        x = cuda_extension.custom_softmax(x, 1)
        yield 'step_003', x
        yield 'step_004', x
