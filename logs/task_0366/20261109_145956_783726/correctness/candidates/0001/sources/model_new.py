import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        equal = cuda_extension.custom_equal(x, y)
        sin = cuda_extension.custom_sin(x)
        return equal, sin

    def debug_forward(self, x: torch.Tensor, y: torch.Tensor):
        equal = cuda_extension.custom_equal(x, y)
        yield 'step_001', equal
        sin = cuda_extension.custom_sin(x)
        yield 'step_002', sin
        yield 'step_003', (equal, sin)
