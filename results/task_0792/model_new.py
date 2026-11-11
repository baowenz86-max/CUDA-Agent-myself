import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.abs_forward(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.abs_forward(x)
        yield 'step_002', x
        yield 'step_003', x


batch_size = 256
width = 1024
height = 1024


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []
