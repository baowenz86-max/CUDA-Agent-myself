import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.softsign_forward(x)

    def debug_forward(self, x: torch.Tensor):
        y = cuda_extension.softsign_forward(x)
        yield 'step_001', y


def get_inputs():
    x = torch.randn(16, 3, 2048, 2048)
    return [x]


def get_init_inputs():
    return []
