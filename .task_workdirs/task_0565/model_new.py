import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = cuda_extension.custom_abs_forward(x)
        x = cuda_extension.custom_clamp_forward(x, self.min_value, self.max_value)
        return x

    def debug_forward(self, x):
        x_abs = cuda_extension.custom_abs_forward(x)
        yield 'step_001', x_abs
        x_clamp = cuda_extension.custom_clamp_forward(x_abs, self.min_value, self.max_value)
        yield 'step_002', x_clamp
        yield 'step_003', x_clamp
