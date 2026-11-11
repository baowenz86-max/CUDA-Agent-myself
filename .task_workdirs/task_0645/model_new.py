import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.sum_dim_forward(x, dim=1)
        x = self.identity(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        y = cuda_extension.sum_dim_forward(x, dim=1)
        yield "step_001", y
        y2 = self.identity(y)
        yield "step_002", y2
        yield "step_003", y2
