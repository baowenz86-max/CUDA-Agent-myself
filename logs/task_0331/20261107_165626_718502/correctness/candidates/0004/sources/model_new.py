import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, bias_shape):
        super().__init__()
        self.fc = nn.Linear(in_features, in_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)
        x = cuda_extension.gelu_forward(x)
        x = cuda_extension.gt_scalar_forward(x, 0.5)
        x = cuda_extension.bool_add_bias_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.linear_forward(x, self.fc.weight, self.fc.bias)
        yield 'step_001', x
        x = cuda_extension.gelu_forward(x)
        yield 'step_002', x
        threshold = torch.tensor(0.5)
        yield 'step_003', threshold
        x = cuda_extension.gt_scalar_forward(x, 0.5)
        yield 'step_004', x
        x = cuda_extension.bool_add_bias_forward(x, self.bias)
        yield 'step_005', x
        yield 'step_006', x
