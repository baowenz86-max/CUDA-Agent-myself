import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, parameter_dict_size, scale_factor, mode, upsample_kernel_size, upsample_padding):
        super().__init__()
        self.parameter_dict = nn.ParameterDict(parameter_dict_size)
        # Store other arguments to match reference attributes
        self.upsample_kernel_size = upsample_kernel_size
        self.upsample_padding = upsample_padding
        # Note: scale_factor and mode are not stored as attributes in reference, but might be needed;
        # we keep them as instance variables without nn.Module to avoid state_dict changes.
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = cuda_extension.custom_upsample_bilinear2d(x, self.scale_factor, False)
        x = cuda_extension.custom_logical_and_gt_lt(x, 0.0, 1.0)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_upsample_bilinear2d(x, self.scale_factor, False)
        yield 'step_001', x
        x = cuda_extension.custom_logical_and_gt_lt(x, 0.0, 1.0)
        yield 'step_002', x
        yield 'step_003', x
        return x
