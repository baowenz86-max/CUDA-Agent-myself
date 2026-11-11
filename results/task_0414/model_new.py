import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = cuda_extension.interpolate_bicubic_forward(x, float(self.scale_factor))
        return x

    def debug_forward(self, x):
        y = cuda_extension.interpolate_bicubic_forward(x, float(self.scale_factor))
        yield 'step_001', y
        yield 'step_002', y
