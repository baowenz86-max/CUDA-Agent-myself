import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, parameter_dict_size, scale_factor, mode, upsample_kernel_size, upsample_padding):
        super().__init__()
        self.parameter_dict = nn.ParameterDict(parameter_dict_size)
        self.scale_factor = scale_factor
        self.mode = mode
        self.upsample_kernel_size = upsample_kernel_size
        self.upsample_padding = upsample_padding

    def forward(self, x):
        x = cuda_extension.upsample_bilinear2d(x, self.scale_factor)
        x = cuda_extension.logical_and_positive_less_than_one(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.upsample_bilinear2d(x, self.scale_factor)
        yield 'step_001', x
        x = cuda_extension.logical_and_positive_less_than_one(x)
        yield 'step_002', x
        yield 'step_003', x
