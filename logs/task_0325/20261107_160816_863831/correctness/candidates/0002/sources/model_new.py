import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.softplus_forward(x)
        x = cuda_extension.relu_forward(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.softplus_forward(x)
        yield 'step_001', x
        x = cuda_extension.relu_forward(x)
        yield 'step_002', x
        yield 'step_003', x


batch_size = 64
height = 2048
width = 2048


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []
