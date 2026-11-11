import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_size):
        super().__init__()
        self.pad_size = pad_size

    def forward(self, x):
        left, right, top, bottom = self.pad_size
        x = cuda_extension.custom_pad(x, left, right, top, bottom)
        return cuda_extension.custom_max_dim(x, 1)

    def debug_forward(self, x):
        left, right, top, bottom = self.pad_size
        padded = cuda_extension.custom_pad(x, left, right, top, bottom)
        yield 'step_001', padded
        max_vals = cuda_extension.custom_max_dim(padded, 1)
        yield 'step_002', max_vals
