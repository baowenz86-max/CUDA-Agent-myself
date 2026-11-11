import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.sign_forward(x)

    def debug_forward(self, x: torch.Tensor):
        y = cuda_extension.sign_forward(x)
        yield 'step_001', y
