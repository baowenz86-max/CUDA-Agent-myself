import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, ndim, axes, k):
        super().__init__()
        self.ndim = ndim
        self.axes = axes
        self.k = k

    def forward(self, x):
        x = cuda_extension.custom_softmax(x, self.ndim)
        x = cuda_extension.custom_rot90(x, self.k, self.axes[0], self.axes[1])
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_softmax(x, self.ndim)
        yield 'step_001', x
        x = cuda_extension.custom_rot90(x, self.k, self.axes[0], self.axes[1])
        yield 'step_002', x
        yield 'step_003', x
