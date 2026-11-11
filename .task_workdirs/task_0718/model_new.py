import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, fmod_divisor):
        super(ModelNew, self).__init__()
        self.fmod_divisor = fmod_divisor

    def forward(self, x):
        x = cuda_extension.custom_sum_dim1(x)
        x = cuda_extension.custom_fmod(x, self.fmod_divisor)
        return x

    def debug_forward(self, x):
        y1 = cuda_extension.custom_sum_dim1(x)
        yield ('step_001', y1)
        y2 = cuda_extension.custom_fmod(y1, self.fmod_divisor)
        yield ('step_002', y2)
        yield ('step_003', y2)
