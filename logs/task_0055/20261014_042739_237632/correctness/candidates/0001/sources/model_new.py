import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.hardtanh_forward(x, -1.0, 1.0)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.hardtanh_forward(x, -1.0, 1.0)
        yield 'step_002', x
        yield 'step_003', x


batch_size = 512
height = 768
width = 768


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []
