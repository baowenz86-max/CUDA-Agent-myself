import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.pool_kernel_size = kernel_size
        self.pool_stride = stride
        self.pool_padding = padding
        self.pool_dilation = dilation
        self.pool_ceil_mode = False

    def forward(self, x):
        x = x.cuda()
        x = cuda_extension.custom_softmax_dim1(x)
        x = cuda_extension.custom_max_pool1d(
            x,
            kernel_size=self.pool_kernel_size,
            stride=self.pool_stride,
            padding=self.pool_padding,
            dilation=self.pool_dilation,
            ceil_mode=self.pool_ceil_mode,
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = x.cuda()
        x = cuda_extension.custom_softmax_dim1(x)
        yield "step_001", x
        x = cuda_extension.custom_max_pool1d(
            x,
            kernel_size=self.pool_kernel_size,
            stride=self.pool_stride,
            padding=self.pool_padding,
            dilation=self.pool_dilation,
            ceil_mode=self.pool_ceil_mode,
        )
        yield "step_002", x
        yield "step_003", x
