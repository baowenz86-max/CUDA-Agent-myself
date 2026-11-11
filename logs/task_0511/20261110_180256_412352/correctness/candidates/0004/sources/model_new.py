import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, parameter_shape):
        super().__init__()
        self.parameter = nn.Parameter(torch.randn(*parameter_shape))

    def forward(self, x):
        return cuda_extension.add_parameter(x, self.parameter)

    def debug_forward(self, x):
        result = cuda_extension.add_parameter(x, self.parameter)
        yield 'step_001', result
