import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.abs_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.abs_forward(x)
        yield 'step_002', x
        yield 'step_003', x


batch_size = 80
height, width = 1024, 1024


def get_inputs():
    x = torch.randn(batch_size, 3, height, width)
    return [x]


def get_init_inputs():
    return []
