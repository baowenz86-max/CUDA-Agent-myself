import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.conv2d_forward(x, self.conv.weight, None)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.conv2d_forward(x, self.conv.weight, None)
        yield 'step_002', x
        yield 'step_003', x
