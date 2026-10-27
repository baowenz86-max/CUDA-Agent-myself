import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, clamp_min, clamp_max, power_factor):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.power_factor = power_factor

    def forward(self, x):
        x = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        x = cuda_extension.pow_forward(x, self.power_factor)
        return x

    def debug_forward(self, x):
        x = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        yield 'step_001', x
        x = cuda_extension.pow_forward(x, self.power_factor)
        yield 'step_002', x
        yield 'step_003', x
