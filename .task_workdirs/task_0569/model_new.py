import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.custom_abs_flat_forward(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.custom_abs_flat_forward(x)
        yield "step_001", x
        yield "step_002", x
        yield "step_003", x
