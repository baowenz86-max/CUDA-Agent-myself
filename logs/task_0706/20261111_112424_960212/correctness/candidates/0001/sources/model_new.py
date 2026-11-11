import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # Original Model stores only AvgPool2d; we keep no parameters, but store config for custom op
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Step 1: sum over channel dimension, keepdim=True
        x = cuda_extension.channel_sum(x)
        # Step 2: average pooling with kernel_size, stride, padding
        x = cuda_extension.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x

    def debug_forward(self, x):
        # Must yield checkpoint ID and value for each reference step, including final return
        # Checkpoint step_001: after channel sum
        x_sum = cuda_extension.channel_sum(x)
        yield 'step_001', x_sum
        # Checkpoint step_002: after pooling
        x_pool = cuda_extension.avg_pool2d(x_sum, self.kernel_size, self.stride, self.padding)
        yield 'step_002', x_pool
        # Checkpoint step_003: final return
        yield 'step_003', x_pool
