import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, num_groups, epsilon):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=epsilon)
        self.num_groups = num_groups
        self.epsilon = epsilon

    def forward(self, x):
        x = cuda_extension.replace_inf_with_nan(x)
        x = cuda_extension.group_norm_forward(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.epsilon)
        return x

    def debug_forward(self, x):
        x = cuda_extension.replace_inf_with_nan(x)
        yield 'step_001', x
        x = cuda_extension.group_norm_forward(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.epsilon)
        yield 'step_002', x
        yield 'step_003', x
