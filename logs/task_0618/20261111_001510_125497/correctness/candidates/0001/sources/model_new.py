import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.maxpool2d_forward(x)
        x = cuda_extension.softmax_dim1_forward(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.maxpool2d_forward(x)
        yield 'step_001', x
        x = cuda_extension.softmax_dim1_forward(x)
        yield 'step_002', x
        yield 'step_003', x
