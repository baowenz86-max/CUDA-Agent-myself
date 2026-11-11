import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.gelu_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.gelu_forward(x)
        yield 'step_001', x
        yield 'step_002', x


batch_size = 16
height, width = 512, 256


def get_inputs():
    x = torch.randn(batch_size, 256, height, width)
    return [x]


def get_init_inputs():
    return []
