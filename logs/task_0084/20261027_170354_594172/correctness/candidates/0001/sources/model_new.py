import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.conv2d_forward(
            x, self.conv.weight, self.conv.bias,
            stride=1, padding=0, dilation=1, groups=1
        )
        x = cuda_extension.elu_add_forward(x, self.bias, alpha=1.0)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv2d_forward(
            x, self.conv.weight, self.conv.bias,
            stride=1, padding=0, dilation=1, groups=1
        )
        yield 'step_001', x
        x = cuda_extension.elu_add_forward(x, self.bias, alpha=1.0)
        yield 'step_002', x
        yield 'step_003', x
