import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.cuda_extension.custom_tanh(x)

    def debug_forward(self, x: torch.Tensor):
        y = torch.ops.cuda_extension.custom_tanh(x)
        yield 'step_001', y
