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

    def forward(self, x):
        x = cuda_extension.hardtanh(x, min_val=-1.0, max_val=1.0)
        x = cuda_extension.avg_pool3d(x, self.kernel_size[0], self.stride[0], self.padding[0])
        return x

    def debug_forward(self, x):
        x = cuda_extension.hardtanh(x, min_val=-1.0, max_val=1.0)
        yield 'step_001', x
        x = cuda_extension.avg_pool3d(x, self.kernel_size[0], self.stride[0], self.padding[0])
        yield 'step_002', x
        yield 'step_003', x
