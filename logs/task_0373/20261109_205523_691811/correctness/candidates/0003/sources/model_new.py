import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        x_max_result = cuda_extension.custom_max_dim1(x)
        x_max, _ = x_max_result
        x_stacked = cuda_extension.custom_stack_dim1(x)
        return x_max, x_stacked

    def debug_forward(self, x):
        x_max_result = cuda_extension.custom_max_dim1(x)
        yield 'step_001', x_max_result
        x_stacked = cuda_extension.custom_stack_dim1(x)
        yield 'step_002', x_stacked
        yield 'step_003', (x_max_result.values, x_stacked)
