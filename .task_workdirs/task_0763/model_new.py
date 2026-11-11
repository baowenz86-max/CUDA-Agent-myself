import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.sum_dim_forward(x, self.dim)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.sum_dim_forward(x, self.dim)
        yield 'step_002', x
        yield 'step_003', x
