import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import cuda_extension
        x = cuda_extension.mean_dim0_keepdim_forward(x)
        x = cuda_extension.group_norm_forward(x, self.num_groups)
        return x

    def debug_forward(self, x: torch.Tensor):
        import cuda_extension
        x = cuda_extension.mean_dim0_keepdim_forward(x)
        yield 'step_001', x
        x = cuda_extension.group_norm_forward(x, self.num_groups)
        yield 'step_002', x
        yield 'step_003', x
