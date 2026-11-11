import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_sum = cuda_extension.reduce_sum_dim(x, self.dim)
        x_nonzero_count = cuda_extension.count_nonzero_dim(x, self.dim)
        return x_sum, x_nonzero_count

    def debug_forward(self, *inputs):
        x = inputs[0]
        x_sum = cuda_extension.reduce_sum_dim(x, self.dim)
        yield 'step_001', x_sum
        x_nonzero_count = cuda_extension.count_nonzero_dim(x, self.dim)
        yield 'step_002', x_nonzero_count
        yield 'step_003', (x_sum, x_nonzero_count)
