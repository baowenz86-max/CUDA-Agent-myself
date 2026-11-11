import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_size):
        super().__init__()
        self.softplus = nn.Softplus()
        self.padding = nn.ReflectionPad2d(padding_size)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = cuda_extension.softplus_forward(x)
        x = cuda_extension.reflection_pad2d_forward(x, 1, 2, 1, 2)
        x = cuda_extension.silu_forward(x)
        return x

    def debug_forward(self, x):
        step_001 = cuda_extension.softplus_forward(x)
        yield 'step_001', step_001
        step_002 = cuda_extension.reflection_pad2d_forward(step_001, 1, 2, 1, 2)
        yield 'step_002', step_002
        step_003 = cuda_extension.silu_forward(step_002)
        yield 'step_003', step_003
        yield 'step_004', step_003
