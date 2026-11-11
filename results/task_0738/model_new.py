import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = cuda_extension.custom_stack(x, y)
        x = cuda_extension.custom_clone(x)
        return x

    def debug_forward(self, *inputs):
        if len(inputs) != 2:
            raise ValueError('Expected exactly two inputs')
        x, y = inputs
        stacked = cuda_extension.custom_stack(x, y)
        yield ('step_001', stacked)
        cloned = cuda_extension.custom_clone(stacked)
        yield ('step_002', cloned)
        yield ('step_003', cloned)


batch_size = 512
channels = 3
height, width = 256, 256


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)
    return [x, y]


def get_init_inputs():
    return []
