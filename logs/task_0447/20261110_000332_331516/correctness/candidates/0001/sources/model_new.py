import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_size, bias_shape):
        super().__init__()
        self.pad_size = tuple(pad_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.constant_pad3d(x, self.pad_size)
        x = cuda_extension.bias_add(x, self.bias)
        return x

    def debug_forward(self, x):
        x_pad = cuda_extension.constant_pad3d(x, self.pad_size)
        yield "step_001", x_pad
        x_out = cuda_extension.bias_add(x_pad, self.bias)
        yield "step_002", x_out
        yield "step_003", x_out
