import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        x = cuda_extension.custom_unfold(
            x,
            self.kernel_size,
            self.dilation,
            self.padding[0],
            self.padding[1],
            self.stride
        )
        x = cuda_extension.custom_pad1d(
            x,
            self.padding[0],
            self.padding[1]
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_unfold(
            x,
            self.kernel_size,
            self.dilation,
            self.padding[0],
            self.padding[1],
            self.stride
        )
        yield ('step_001', x)
        x = cuda_extension.custom_pad1d(
            x,
            self.padding[0],
            self.padding[1]
        )
        yield ('step_002', x)
        yield ('step_003', x)
