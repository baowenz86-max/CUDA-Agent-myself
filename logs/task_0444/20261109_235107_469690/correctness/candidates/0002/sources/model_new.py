import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.custom_sigmoid(x)
        x = cuda_extension.custom_prod_dim1(x)
        x = cuda_extension.custom_clamp(x, 0.0, 1.0)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_sigmoid(x)
        yield 'step_001', x
        x = cuda_extension.custom_prod_dim1(x)
        yield 'step_002', x
        x = cuda_extension.custom_clamp(x, 0.0, 1.0)
        yield 'step_003', x
        yield 'step_004', x