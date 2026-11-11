import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, padding, input_shape):
        super().__init__()
        self.padding = padding
        self.input_shape = input_shape

    def forward(self, x):
        x = cuda_extension.custom_acos(x)
        x = cuda_extension.custom_reflection_pad2d(x, self.padding)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_acos(x)
        yield 'step_001', x
        x = cuda_extension.custom_reflection_pad2d(x, self.padding)
        yield 'step_002', x
        yield 'step_003', x
