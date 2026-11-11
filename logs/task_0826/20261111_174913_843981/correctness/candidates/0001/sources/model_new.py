import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.rot90_clone_forward(x)

    def debug_forward(self, x: torch.Tensor):
        x_clone = cuda_extension.rot90_clone_forward(x)
        yield 'step_001', x_clone
        x_rotated = cuda_extension.rot90_clone_forward(x_clone)
        yield 'step_002', x_rotated
        yield 'step_003', x_rotated
