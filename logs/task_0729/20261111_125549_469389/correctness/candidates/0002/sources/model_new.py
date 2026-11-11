import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, reduction_dim: tuple):
        super().__init__()
        self.reduction_dim = reduction_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom op for torch.sum over reduction_dim
        # reduction_dim is expected to be a tuple of ints, e.g., (2,3)
        # We assume it has exactly two elements; if more, we need different handling.
        # But task contract shows (2,3). We'll use two int parameters dim0, dim1.
        dim0, dim1 = self.reduction_dim
        x = cuda_extension.custom_reduce_sum(x, dim0, dim1)
        # Custom op for torch.round
        x = cuda_extension.custom_round(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        # Same computations but yield intermediate checkpoints
        dim0, dim1 = self.reduction_dim
        x_sum = cuda_extension.custom_reduce_sum(x, dim0, dim1)
        yield 'step_001', x_sum
        x_round = cuda_extension.custom_round(x_sum)
        yield 'step_002', x_round
        yield 'step_003', x_round
