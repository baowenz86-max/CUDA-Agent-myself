import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = cuda_extension.upsample_nearest2d(x, self.scale_factor)
        x = cuda_extension.relu(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.upsample_nearest2d(x, self.scale_factor)
        yield 'step_001', x
        x = cuda_extension.relu(x)
        yield 'step_002', x
        yield 'step_003', x
