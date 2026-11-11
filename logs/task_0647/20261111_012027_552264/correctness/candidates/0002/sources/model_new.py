import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, batch_size, height, width):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

    def forward(self, x):
        x = cuda_extension.custom_unfold_abs(x, kernel_h=self.height, kernel_w=self.width)
        return x

    def debug_forward(self, x):
        x_unfold = cuda_extension.custom_unfold(x, kernel_h=self.height, kernel_w=self.width)
        yield 'step_001', x_unfold
        x_abs = cuda_extension.custom_abs(x_unfold)
        yield 'step_002', x_abs
        yield 'step_003', x_abs
