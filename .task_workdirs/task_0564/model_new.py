import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.max_dim1(x)
        x = cuda_extension.log1p(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.max_dim1(x)
        yield ('step_001', x)
        x = cuda_extension.log1p(x)
        yield ('step_002', x)
        yield ('step_003', x)
