import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode, padding_dim, padding_value):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.padding_dim = padding_dim
        self.padding_value = padding_value

    def forward(self, x):
        x = cuda_extension.upsampling_bilinear2d_forward(x, float(self.scale_factor), False)
        left, right, top, bottom = self.padding_dim
        x = cuda_extension.constant_pad2d_forward(x, [left, right, top, bottom], float(self.padding_value))
        return x

    def debug_forward(self, x):
        upsampled = cuda_extension.upsampling_bilinear2d_forward(x, float(self.scale_factor), False)
        yield 'step_001', upsampled
        left, right, top, bottom = self.padding_dim
        padded = cuda_extension.constant_pad2d_forward(upsampled, [left, right, top, bottom], float(self.padding_value))
        yield 'step_002', padded
        yield 'step_003', padded
