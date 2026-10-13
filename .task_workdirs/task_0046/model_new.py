import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor=1.0, mode='bilinear'):
        super(ModelNew, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = cuda_extension.interpolate_bilinear_scale(x, self.scale_factor)
        x = cuda_extension.relu_where(x)
        x = cuda_extension.interpolate_bilinear_scale(x, self.scale_factor)
        return x

    def debug_forward(self, x):
        x = cuda_extension.interpolate_bilinear_scale(x, self.scale_factor)
        yield 'step_001', x
        x = cuda_extension.relu_where(x)
        yield 'step_002', x
        x = cuda_extension.interpolate_bilinear_scale(x, self.scale_factor)
        yield 'step_003', x
        yield 'step_004', x
