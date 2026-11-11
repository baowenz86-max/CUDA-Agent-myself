import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_parameters, use_relu):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters, use_relu)

    def forward(self, x):
        x = cuda_extension.abs_forward(x)
        x = cuda_extension.prelu_forward(x, self.prelu.weight)
        return x

    def debug_forward(self, x):
        x_abs = cuda_extension.abs_forward(x)
        yield 'step_001', x_abs
        x_prelu = cuda_extension.prelu_forward(x_abs, self.prelu.weight)
        yield 'step_002', x_prelu
        yield 'step_003', x_prelu
