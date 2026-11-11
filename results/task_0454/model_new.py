import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, pooling_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.pooling_size = pooling_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.custom_abs_forward(x)
        x = cuda_extension.custom_adaptive_avg_pool1d_forward(x, self.pooling_size)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.custom_abs_forward(x)
        yield 'step_001', x
        x = cuda_extension.custom_adaptive_avg_pool1d_forward(x, self.pooling_size)
        yield 'step_002', x
        yield 'step_003', x
