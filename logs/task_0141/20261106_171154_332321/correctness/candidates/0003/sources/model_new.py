import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = x.to('cuda')
        x = cuda_extension.leaky_relu_forward(x, negative_slope=0.01)
        x = cuda_extension.max_pool2d_forward(x, kernel_size=2, stride=2)
        x = cuda_extension.conv2d_forward(x, self.conv2d.weight, self.conv2d.bias, stride=1, dilation=1)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0].to('cuda')
        x = cuda_extension.leaky_relu_forward(x, negative_slope=0.01)
        yield 'step_001', x
        x = cuda_extension.max_pool2d_forward(x, kernel_size=2, stride=2)
        yield 'step_002', x
        x = cuda_extension.conv2d_forward(x, self.conv2d.weight, self.conv2d.bias, stride=1, dilation=1)
        yield 'step_003', x
        yield 'step_004', x
