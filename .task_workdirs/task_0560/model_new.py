import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        # Reference Model does not store any parameters or buffers; only submodules.
        # We store configuration for use in custom ops.
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        x = cuda_extension.softplus_forward(x)
        x = cuda_extension.avgpool2d_forward(x, self.pool_kernel_size, self.pool_stride, self.pool_padding)
        return x

    def debug_forward(self, *inputs):
        if len(inputs) != 1:
            raise ValueError("Expected one input tensor")
        x = inputs[0]
        # checkpoint step_001: softplus output
        x = cuda_extension.softplus_forward(x)
        yield "step_001", x
        # checkpoint step_002: avg_pool output
        x = cuda_extension.avgpool2d_forward(x, self.pool_kernel_size, self.pool_stride, self.pool_padding)
        yield "step_002", x
        # checkpoint step_003: final return
        yield "step_003", x
