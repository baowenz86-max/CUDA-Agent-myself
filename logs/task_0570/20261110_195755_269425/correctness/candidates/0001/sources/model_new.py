import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x, y):
        # Apply ReflectionPad2d using custom CUDA operator
        x = cuda_extension.reflection_pad2d_uint8(x, self.padding)
        # Apply bitwise XOR using custom CUDA operator
        x = cuda_extension.bitwise_xor_uint8(x, y)
        return x

    def debug_forward(self, *inputs):
        x, y = inputs
        # checkpoint step_001: input after reflection padding
        x_padded = cuda_extension.reflection_pad2d_uint8(x, self.padding)
        yield 'step_001', x_padded
        # checkpoint step_002: after bitwise xor
        x_out = cuda_extension.bitwise_xor_uint8(x_padded, y)
        yield 'step_002', x_out
        # checkpoint step_003: final return
        yield 'step_003', x_out
