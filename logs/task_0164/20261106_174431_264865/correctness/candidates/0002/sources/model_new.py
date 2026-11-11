import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.ceil_mode = False

    def forward(self, x):
        x = cuda_extension.maxpool3d_forward(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )
        x = cuda_extension.prod_dim_forward(x, 1, True)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.maxpool3d_forward(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )
        yield 'step_001', x1
        x2 = cuda_extension.prod_dim_forward(x1, 1, True)
        yield 'step_002', x2
        yield 'step_003', x2
