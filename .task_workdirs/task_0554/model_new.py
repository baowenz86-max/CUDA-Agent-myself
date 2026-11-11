import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pool_size, stride, dilation, return_indices=False):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x):
        kernel_size = self.pool_size[0] if isinstance(self.pool_size, tuple) else self.pool_size
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]
        padding = 0
        dilation = self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        ceil_mode = False
        x = cuda_extension.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
        return x

    def debug_forward(self, x):
        kernel_size = self.pool_size[0] if isinstance(self.pool_size, tuple) else self.pool_size
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]
        padding = 0
        dilation = self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        ceil_mode = False
        x = cuda_extension.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
        yield "step_001", x
        yield "step_002", x
