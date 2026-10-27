import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = cuda_extension.custom_upsample_bilinear2d(x, self.scale_factor)
        x = cuda_extension.custom_relu(x)
        return x

    def debug_forward(self, x):
        x_upsampled = cuda_extension.custom_upsample_bilinear2d(x, self.scale_factor)
        yield 'step_001', x_upsampled
        x_relu = cuda_extension.custom_relu(x_upsampled)
        yield 'step_002', x_relu
        yield 'step_003', x_relu
