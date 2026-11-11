import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_size, bias_shape):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.pad.padding)
        x = cuda_extension.sigmoid_forward(x)
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        x = cuda_extension.add_bias_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.pad.padding)
        yield 'step_001', x
        x = cuda_extension.sigmoid_forward(x)
        yield 'step_002', x
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        yield 'step_003', x
        x = cuda_extension.add_bias_forward(x, self.bias)
        yield 'step_004', x
        yield 'step_005', x
