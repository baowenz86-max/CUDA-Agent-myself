import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_size):
        super().__init__()
        # padding_size is a tuple of 4 ints (left, right, top, bottom)
        self.padding_size = padding_size

    def forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.padding_size)
        x = cuda_extension.ge_scalar_forward(x, 0)
        x = cuda_extension.remainder_scalar_forward(x, 2)
        return x

    def debug_forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.padding_size)
        yield 'step_001', x
        x = cuda_extension.ge_scalar_forward(x, 0)
        yield 'step_002', x
        x = cuda_extension.remainder_scalar_forward(x, 2)
        yield 'step_003', x
        yield 'step_004', x