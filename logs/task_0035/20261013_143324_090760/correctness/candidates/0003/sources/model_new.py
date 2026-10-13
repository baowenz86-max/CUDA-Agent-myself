import torch
import torch.nn as nn
import cuda_extension


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
        x = cuda_extension.logical_and_range(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        step_001 = cuda_extension.upsample_bilinear2d(x, self.scale_factor)
        yield ('step_001', step_001)
        step_002 = cuda_extension.logical_and_range(step_001)
        yield ('step_002', step_002)
        yield ('step_003', step_002)
