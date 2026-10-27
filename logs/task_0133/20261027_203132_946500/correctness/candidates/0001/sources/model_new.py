import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_size, size_before, size_after):
        super().__init__()
        self.pad_size = pad_size
        self.size_after = size_after

    def forward(self, x):
        x = cuda_extension.constant_pad1d(x, self.pad_size[0], self.pad_size[1], 0.0)
        x = cuda_extension.interpolate_nearest(x, self.size_after[0], self.size_after[1])
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.constant_pad1d(x, self.pad_size[0], self.pad_size[1], 0.0)
        yield ('step_001', x)
        x = cuda_extension.interpolate_nearest(x, self.size_after[0], self.size_after[1])
        yield ('step_002', x)
        yield ('step_003', x)
