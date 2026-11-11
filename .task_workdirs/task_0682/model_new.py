import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.mean_dim0(x)
        x = cuda_extension.view_as_complex_lastdim(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.mean_dim0(x)
        yield 'step_001', x
        x = cuda_extension.view_as_complex_lastdim(x)
        yield 'step_002', x
        yield 'step_003', x
