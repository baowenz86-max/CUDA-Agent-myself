import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.triu(x)
        x = cuda_extension.conv2d(x, self.conv.weight, self.conv.bias)
        x = cuda_extension.bias_add(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.triu(x)
        yield 'step_001', x
        x = cuda_extension.conv2d(x, self.conv.weight, self.conv.bias)
        yield 'step_002', x
        x = cuda_extension.bias_add(x, self.bias)
        yield 'step_003', x
        yield 'step_004', x
