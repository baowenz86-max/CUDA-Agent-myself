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
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias,
                                          self.conv.stride, self.conv.padding)
        x = cuda_extension.add_bias_forward(x, self.bias)
        x = cuda_extension.batch_norm_forward(x, self.bn.weight, self.bn.bias,
                                               self.bn.running_mean, self.bn.running_var,
                                               self.bn.eps)
        x = cuda_extension.avg_pool2d_forward(x, self.pool.kernel_size,
                                              self.pool.stride, self.pool.padding)
        x = cuda_extension.sigmoid_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias,
                                          self.conv.stride, self.conv.padding)
        yield 'step_001', x
        x = cuda_extension.add_bias_forward(x, self.bias)
        yield 'step_002', x
        x = cuda_extension.batch_norm_forward(x, self.bn.weight, self.bn.bias,
                                               self.bn.running_mean, self.bn.running_var,
                                               self.bn.eps)
        yield 'step_003', x
        x = cuda_extension.avg_pool2d_forward(x, self.pool.kernel_size,
                                              self.pool.stride, self.pool.padding)
        yield 'step_004', x
        x = cuda_extension.sigmoid_forward(x)
        yield 'step_005', x
        yield 'step_006', x
