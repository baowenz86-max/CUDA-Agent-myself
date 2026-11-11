import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, num_groups, num_features):
        super().__init__()
        # Match original state_dict keys: group_norm.weight, group_norm.bias
        self.group_norm = nn.Module()
        self.group_norm.weight = nn.Parameter(torch.ones(num_channels))
        self.group_norm.bias = nn.Parameter(torch.zeros(num_channels))
        self.num_groups = num_groups
        self.eps = 1e-05

    def forward(self, x):
        x = cuda_extension.custom_group_norm(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.eps)
        x = cuda_extension.custom_tanh(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_group_norm(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.eps)
        yield ('step_001', x)
        x = cuda_extension.custom_tanh(x)
        yield ('step_002', x)
        yield ('step_003', x)
