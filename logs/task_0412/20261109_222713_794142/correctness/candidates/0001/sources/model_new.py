import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, num_units, kernel_size, stride):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.num_units = num_units

    def forward(self, x):
        x = cuda_extension.custom_batch_norm2d(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.training,
            0.1,
            1e-05,
        )
        x = cuda_extension.custom_avg_pool2d(
            x,
            self.avg_pool.kernel_size,
            self.avg_pool.stride,
            self.avg_pool.padding,
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_batch_norm2d(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.training,
            0.1,
            1e-05,
        )
        yield 'step_001', x
        x = cuda_extension.custom_avg_pool2d(
            x,
            self.avg_pool.kernel_size,
            self.avg_pool.stride,
            self.avg_pool.padding,
        )
        yield 'step_002', x
        yield 'step_003', x
