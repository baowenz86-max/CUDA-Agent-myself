import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_dim):
        super(ModelNew, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        max_val, _ = cuda_extension.custom_max_dim(x, self.input_dim)
        argmin_val = cuda_extension.custom_argmin_dim(x, self.input_dim)
        return max_val, argmin_val

    def debug_forward(self, x):
        max_val, _ = cuda_extension.custom_max_dim(x, self.input_dim)
        yield 'step_001', (max_val, _)
        argmin_val = cuda_extension.custom_argmin_dim(x, self.input_dim)
        yield 'step_002', argmin_val
        yield 'step_003', (max_val, argmin_val)
