import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.custom_mean(x, 0)
        x = cuda_extension.custom_hardswish(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        x_mean = cuda_extension.custom_mean(x, 0)
        yield 'step_001', x_mean
        x_hardswish = cuda_extension.custom_hardswish(x_mean)
        yield 'step_002', x_hardswish
        yield 'step_003', x_hardswish
