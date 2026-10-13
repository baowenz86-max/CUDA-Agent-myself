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
        # Keep upsample module for state_dict compatibility, but do not call it
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        upsampled = cuda_extension.bilinear_upsample(x, self.scale_factor)
        out = cuda_extension.logical_and_threshold(upsampled)
        return out

    def debug_forward(self, x):
        upsampled = cuda_extension.bilinear_upsample(x, self.scale_factor)
        yield 'step_001', upsampled
        out = cuda_extension.logical_and_threshold(upsampled)
        yield 'step_002', out
        yield 'step_003', out
