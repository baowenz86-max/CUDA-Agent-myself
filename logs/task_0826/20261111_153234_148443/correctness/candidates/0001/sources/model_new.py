import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clone = cuda_extension.clone_to_cuda(x)
        x_rotated = cuda_extension.rot90_1_2(x_clone)
        return x_rotated

    def debug_forward(self, *inputs):
        x = inputs[0]
        x_clone = cuda_extension.clone_to_cuda(x)
        yield 'step_001', x_clone
        x_rotated = cuda_extension.rot90_1_2(x_clone)
        yield 'step_002', x_rotated
        yield 'step_003', x_rotated
