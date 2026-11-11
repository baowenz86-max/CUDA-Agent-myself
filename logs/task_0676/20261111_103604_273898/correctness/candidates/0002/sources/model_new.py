import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, reduction_ratio):
        super().__init__()
        self.identity = nn.Identity()
        self.reduction_ratio = reduction_ratio

    def forward(self, x):
        # Apply max operation
        x, _ = cuda_extension.max_reduce_div_forward(
            x, dim=1, keepdim=True, divisor=self.reduction_ratio
        )
        # Apply identity operation
        x = self.identity(x)
        # x = x / self.reduction_ratio  # already applied inside fused op
        return x

    def debug_forward(self, x):
        values, indices = cuda_extension.max_reduce_div_forward(
            x, dim=1, keepdim=True, divisor=1.0
        )
        yield 'step_001', (values, indices)
        identity_out = self.identity(values)
        yield 'step_002', identity_out
        divided = identity_out / self.reduction_ratio
        yield 'step_003', divided
        yield 'step_004', divided
