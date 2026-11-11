import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, padding_size, divisor):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.divisor = divisor

    def forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.reflection_pad.padding[0], self.reflection_pad.padding[1], self.reflection_pad.padding[2], self.reflection_pad.padding[3])
        x = cuda_extension.remainder_scalar_forward(x, self.divisor)
        return x

    def debug_forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.reflection_pad.padding[0], self.reflection_pad.padding[1], self.reflection_pad.padding[2], self.reflection_pad.padding[3])
        yield 'step_001', x
        x = cuda_extension.remainder_scalar_forward(x, self.divisor)
        yield 'step_002', x
        yield 'step_003', x
