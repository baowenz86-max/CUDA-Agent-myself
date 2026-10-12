import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.conv2d_forward(x, self.conv2d.weight, self.conv2d.bias)
        x = cuda_extension.triu_forward(x)
        x_uint8 = x.to(torch.uint8)
        bias_uint8 = self.bias.to(torch.uint8).expand_as(x_uint8)
        x = cuda_extension.bitwise_xor_forward(x_uint8, bias_uint8)
        x = cuda_extension.cast_float_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv2d_forward(x, self.conv2d.weight, self.conv2d.bias)
        yield 'step_001', x
        x = cuda_extension.triu_forward(x)
        yield 'step_002', x
        x_uint8 = x.to(torch.uint8)
        bias_uint8 = self.bias.to(torch.uint8).expand_as(x_uint8)
        x = cuda_extension.bitwise_xor_forward(x_uint8, bias_uint8)
        yield 'step_003', x
        x = cuda_extension.cast_float_forward(x)
        yield 'step_004', x
