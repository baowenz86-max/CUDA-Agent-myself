import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, data_shape, values_shape):
        super().__init__()
        self.data = nn.ParameterList([nn.Parameter(torch.randn(data_shape)) for _ in range(2)])
        self.values = nn.Parameter(torch.randn(values_shape))

    def forward(self, x):
        x = cuda_extension.custom_isin(x, self.values)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        step_001 = cuda_extension.custom_isin(x, self.values)
        yield 'step_001', step_001
        step_002 = step_001
        yield 'step_002', step_002
