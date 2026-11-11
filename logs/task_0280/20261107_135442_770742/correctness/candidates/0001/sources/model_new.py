import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.padding)
        x = cuda_extension.log2_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.padding)
        yield 'step_001', x
        x = cuda_extension.log2_forward(x)
        yield 'step_002', x
        yield 'step_003', x


batch_size = 128
channels = 3
height, width = 1024, 256
padding = 2


def get_inputs():
    return [torch.rand(batch_size, channels, height, width) + 1e-5]


def get_init_inputs():
    return [padding]
