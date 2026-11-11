import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return cuda_extension.custom_cat(x1, x2, self.dim)

    def debug_forward(self, *inputs):
        result = self.forward(*inputs)
        yield 'step_001', result
