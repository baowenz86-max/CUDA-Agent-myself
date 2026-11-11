import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        x = cuda_extension.logsoftmax_forward(x, 1)
        x = cuda_extension.maxpool3d_forward(
            x, self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2],
            self.pool_stride[0], self.pool_stride[1], self.pool_stride[2],
            self.pool_padding[0], self.pool_padding[1], self.pool_padding[2]
        )
        return x

    def debug_forward(self, *inputs):
        x, = inputs
        x = cuda_extension.logsoftmax_forward(x, 1)
        yield 'step_001', x
        x = cuda_extension.maxpool3d_forward(
            x, self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2],
            self.pool_stride[0], self.pool_stride[1], self.pool_stride[2],
            self.pool_padding[0], self.pool_padding[1], self.pool_padding[2]
        )
        yield 'step_002', x
        yield 'step_003', x
