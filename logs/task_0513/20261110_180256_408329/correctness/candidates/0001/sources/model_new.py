import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.maxpool1d_forward(x, self.kernel_size, self.stride, self.padding, False)

    def debug_forward(self, *inputs):
        x = inputs[0]
        out = self.forward(x)
        yield 'step_001', out
