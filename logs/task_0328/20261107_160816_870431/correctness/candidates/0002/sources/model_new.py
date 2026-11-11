import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, pool_kernel_size, pool_stride):
        super(ModelNew, self).__init__()
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.max_pool2d_forward(
            x,
            kernel_size=self.pool_kernel_size,
            stride=self.pool_stride,
            padding=0,
            dilation=1,
            ceil_mode=False)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu_forward(x)
        yield "step_001", x
        x = cuda_extension.max_pool2d_forward(
            x,
            kernel_size=self.pool_kernel_size,
            stride=self.pool_stride,
            padding=0,
            dilation=1,
            ceil_mode=False)
        yield "step_002", x
        yield "step_003", x
