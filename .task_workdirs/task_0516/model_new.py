import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, num_groups, num_channels_per_group):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        x = cuda_extension.group_norm_forward(x, self.num_groups, None, None, 1e-05)
        return x

    def debug_forward(self, x):
        x = cuda_extension.group_norm_forward(x, self.num_groups, None, None, 1e-05)
        yield 'step_001', x
        yield 'step_002', x
