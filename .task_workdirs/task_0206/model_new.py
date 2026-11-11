import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = cuda_extension.leaky_relu_fwd(x, 0.01)
        x = cuda_extension.abs_fwd(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.leaky_relu_fwd(x, 0.01)
        yield 'step_001', x
        x = cuda_extension.abs_fwd(x)
        yield 'step_002', x
        yield 'step_003', x
