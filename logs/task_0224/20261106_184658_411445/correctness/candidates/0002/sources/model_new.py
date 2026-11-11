import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_channels, kernel_size, stride, padding):
        super().__init__()
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = cuda_extension.hardtanh_forward(x, -1.0, 1.0)
        x = cuda_extension.avgpool2d_forward(x, self.kernel_size, self.stride, self.padding)
        return x

    def debug_forward(self, x):
        x = cuda_extension.hardtanh_forward(x, -1.0, 1.0)
        yield ('step_001', x)
        x = cuda_extension.avgpool2d_forward(x, self.kernel_size, self.stride, self.padding)
        yield ('step_002', x)
        yield ('step_003', x)
