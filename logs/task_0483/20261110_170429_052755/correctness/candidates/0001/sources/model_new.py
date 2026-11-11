import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.out_channels = out_channels
        self.avgpool_out_size = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.adaptive_avg_pool1d_transpose(x, self.avgpool_out_size)

    def debug_forward(self, x: torch.Tensor):
        x_t = torch.transpose(x, 1, 2)
        yield 'step_001', x_t
        x_pool = cuda_extension.adaptive_avg_pool1d_transpose(x, self.avgpool_out_size)
        yield 'step_002', x_pool
        yield 'step_003', x_pool
