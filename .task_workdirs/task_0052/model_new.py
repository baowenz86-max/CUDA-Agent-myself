import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, interp_scale_factor, interp_mode, argmax_dim):
        super().__init__()
        self.in_channels = in_channels
        self.interp_scale_factor = interp_scale_factor
        self.interp_mode = interp_mode
        self.argmax_dim = argmax_dim

    def forward(self, x):
        x = cuda_extension.custom_round(x)
        x = cuda_extension.custom_interpolate(x, self.interp_scale_factor, 0)  # mode encoded as int: 0 for bilinear
        x = cuda_extension.custom_argmax(x, self.argmax_dim)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_round(x)
        yield "step_001", x
        x = cuda_extension.custom_interpolate(x, self.interp_scale_factor, 0)
        yield "step_002", x
        x = cuda_extension.custom_argmax(x, self.argmax_dim)
        yield "step_003", x
        yield "step_004", x
