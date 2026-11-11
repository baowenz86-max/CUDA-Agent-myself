import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_value):
        super().__init__()
        self.pad_value = pad_value

    def forward(self, x):
        x = cuda_extension.constant_pad(x, pad=1, value=float(self.pad_value))
        x = cuda_extension.stack_duplicate(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.constant_pad(x, pad=1, value=float(self.pad_value))
        yield 'step_001', x
        x = cuda_extension.stack_duplicate(x)
        yield 'step_002', x
        yield 'step_003', x
