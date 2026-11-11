import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = cuda_extension.avgpool2d_forward(x, self.kernel_size, self.stride, self.padding)
        x = cuda_extension.tanh_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.avgpool2d_forward(x, self.kernel_size, self.stride, self.padding)
        yield "step_001", x
        x = cuda_extension.tanh_forward(x)
        yield "step_002", x
        yield "step_003", x
