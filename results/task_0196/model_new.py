import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.gelu_forward(x)
        x = cuda_extension.tanh_forward(x)
        return x

    def debug_forward(self, x):
        gelu_out = cuda_extension.gelu_forward(x)
        yield 'step_001', gelu_out
        tanh_out = cuda_extension.tanh_forward(gelu_out)
        yield 'step_002', tanh_out
        yield 'step_003', tanh_out
