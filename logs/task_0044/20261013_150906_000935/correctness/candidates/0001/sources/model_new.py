import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.cuda()
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias, 1, 1, 0, 0, 1, 1)
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        x = cuda_extension.add_bias_forward(x, self.bias)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0].cuda()
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias, 1, 1, 0, 0, 1, 1)
        yield 'step_001', x
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        yield 'step_002', x
        x = cuda_extension.add_bias_forward(x, self.bias)
        yield 'step_003', x
        yield 'step_004', x
