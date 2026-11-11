import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.avgpool2d_forward(x, self.kernel_size, self.stride)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.avgpool2d_forward(x, self.kernel_size, self.stride)
        yield 'step_002', x
        yield 'step_003', x
