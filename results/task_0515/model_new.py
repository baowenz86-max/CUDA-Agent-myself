import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, ceil_mode):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(kernel_size)
        self.linear = nn.Linear(in_channels * kernel_size[0] * kernel_size[1], out_channels)

    def forward(self, x):
        x = cuda_extension.adaptive_avg_pool2d_forward(x, self.adaptive_avg_pool.output_size[0], self.adaptive_avg_pool.output_size[1])
        x = cuda_extension.view_flatten_forward(x, x.size(0))
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.adaptive_avg_pool2d_forward(x, self.adaptive_avg_pool.output_size[0], self.adaptive_avg_pool.output_size[1])
        yield 'step_001', x
        x = cuda_extension.view_flatten_forward(x, x.size(0))
        yield 'step_002', x
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        yield 'step_003', x
        yield 'step_004', x
