import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(in_channels)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = cuda_extension.batch_norm2d_forward(x, self.batchnorm2d.weight, self.batchnorm2d.bias, self.batchnorm2d.running_mean, self.batchnorm2d.running_var, self.batchnorm2d.eps, self.batchnorm2d.momentum, self.training)
        x = cuda_extension.hardswish_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.batch_norm2d_forward(x, self.batchnorm2d.weight, self.batchnorm2d.bias, self.batchnorm2d.running_mean, self.batchnorm2d.running_var, self.batchnorm2d.eps, self.batchnorm2d.momentum, self.training)
        yield 'step_001', x
        x = cuda_extension.hardswish_forward(x)
        yield 'step_002', x
        yield 'step_003', x
