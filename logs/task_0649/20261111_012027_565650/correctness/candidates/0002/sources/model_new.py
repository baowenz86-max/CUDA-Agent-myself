import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x_cat = cuda_extension.concat_forward(x1, x2, self.dim)
        x_amax = cuda_extension.amax_forward(x_cat, self.dim)
        return x_amax

    def debug_forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x_cat = cuda_extension.concat_forward(x1, x2, self.dim)
        yield 'step_001', x_cat
        x_amax = cuda_extension.amax_forward(x_cat, self.dim)
        yield 'step_002', x_amax
        yield 'step_003', x_amax
