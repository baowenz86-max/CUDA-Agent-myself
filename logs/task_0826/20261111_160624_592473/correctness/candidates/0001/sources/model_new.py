import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rotated = cuda_extension.custom_rot90(x)
        return x_rotated

    def debug_forward(self, x: torch.Tensor):
        x_rotated = cuda_extension.custom_rot90(x)
        yield 'step_001', x_rotated
        yield 'step_002', x_rotated
        yield 'step_003', x_rotated
