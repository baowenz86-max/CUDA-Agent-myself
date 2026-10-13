import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, bias_shape, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.conv2d_bias_bn(
            x, self.conv.weight, self.conv.bias, self.bias, self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var, self.bn.eps, self.pool.kernel_size,
            self.pool.stride, self.pool.padding
        )
        x = cuda_extension.avg_pool2d(x, self.pool.kernel_size, self.pool.stride, self.pool.padding)
        x = cuda_extension.sigmoid(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv2d_bias_bn(
            x, self.conv.weight, self.conv.bias, self.bias, self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var, self.bn.eps, self.pool.kernel_size,
            self.pool.stride, self.pool.padding
        )
        yield 'step_001', None
        yield 'step_002', x
        x = cuda_extension.avg_pool2d(x, self.pool.kernel_size, self.pool.stride, self.pool.padding)
        yield 'step_003', x
        x = cuda_extension.sigmoid(x)
        yield 'step_004', x
        yield 'step_005', x
        return x
