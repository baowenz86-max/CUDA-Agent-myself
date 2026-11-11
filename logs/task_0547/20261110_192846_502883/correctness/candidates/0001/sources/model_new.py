import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = cuda_extension.custom_triu(x, diagonal=1)
        x = cuda_extension.custom_abs(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_triu(x, diagonal=1)
        yield 'step_001', x
        x = cuda_extension.custom_abs(x)
        yield 'step_002', x
        yield 'step_003', x
