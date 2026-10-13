import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias_shape = bias_shape

    def forward(self, x):
        x = cuda_extension.conv2d_forward(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=1
        )
        x = cuda_extension.leaky_relu_forward(x, negative_slope=0.01)
        x = cuda_extension.add_bias_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv2d_forward(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=1
        )
        yield 'step_001', x
        x = cuda_extension.leaky_relu_forward(x, negative_slope=0.01)
        yield 'step_002', x
        x = cuda_extension.add_bias_forward(x, self.bias)
        yield 'step_003', x
        yield 'step_004', x
