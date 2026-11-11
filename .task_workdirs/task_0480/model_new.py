import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.softplus_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.softplus_forward(x)
        yield 'step_002', x
        yield 'step_003', x
