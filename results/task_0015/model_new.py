import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_value, scale_factor, mode):
        super().__init__()
        # Use nested module to match reference state_dict key 'prelu.weight'
        self.prelu = nn.PReLU(in_channels, init=alpha_value)
        self.scale_factor = float(scale_factor)
        self.mode = mode

    def forward(self, x):
        # Apply PReLU using custom CUDA kernel and load weight from nested module
        x = cuda_extension.prelu_forward(x, self.prelu.weight)
        if self.mode != 'bilinear':
            raise NotImplementedError(f'Unsupported interpolation mode: {self.mode}')
        x = cuda_extension.bilinear_interpolate_forward(x, self.scale_factor)
        x = cuda_extension.argmax_dim_forward(x, 1, True)
        return x

    def debug_forward(self, x):
        # Produce intermediate checkpoints matching reference trace
        prelu_out = cuda_extension.prelu_forward(x, self.prelu.weight)
        yield 'step_001', prelu_out
        interp_out = cuda_extension.bilinear_interpolate_forward(prelu_out, self.scale_factor)
        yield 'step_002', interp_out
        argmax_out = cuda_extension.argmax_dim_forward(interp_out, 1, True)
        yield 'step_003', argmax_out
        yield 'step_004', argmax_out
