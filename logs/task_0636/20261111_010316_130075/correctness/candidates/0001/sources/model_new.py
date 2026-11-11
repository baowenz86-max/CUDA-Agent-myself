import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.maxpool1d(x, self.kernel_size, self.stride, self.padding)
        x = cuda_extension.tanh(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.maxpool1d(x, self.kernel_size, self.stride, self.padding)
        yield 'step_001', x
        x = cuda_extension.tanh(x)
        yield 'step_002', x
        yield 'step_003', x
