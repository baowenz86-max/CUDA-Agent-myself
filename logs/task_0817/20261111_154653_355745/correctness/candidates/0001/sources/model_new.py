import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        x = cuda_extension.custom_cos(x)
        x = cuda_extension.custom_identity(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_cos(x)
        yield 'step_001', x
        x = cuda_extension.custom_identity(x)
        yield 'step_002', x
        yield 'step_003', x


batch_size = 512
height, width = 1024, 1024


def get_inputs():
    return [torch.randn(batch_size, 1, height, width)]


def get_init_inputs():
    return []
