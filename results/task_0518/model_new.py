import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.softmin_forward(x, self.dim)

    def debug_forward(self, *inputs):
        x = inputs[0]
        out = self.forward(x)
        yield 'step_001', out
