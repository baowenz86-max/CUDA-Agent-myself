import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = cuda_extension.clamp_scalar(x, self.min_value, self.max_value)
        x = cuda_extension.sub_scalar(x, self.min_value)
        return x

    def debug_forward(self, x):
        x_clamped = cuda_extension.clamp_scalar(x, self.min_value, self.max_value)
        yield "step_001", x_clamped
        x_sub = cuda_extension.sub_scalar(x_clamped, self.min_value)
        yield "step_002", x_sub
        yield "step_003", x_sub
