import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, num_groups, eps, affine):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.group_norm_weight = nn.Parameter(torch.ones(num_channels))
            self.group_norm_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('group_norm_weight', None)
            self.register_parameter('group_norm_bias', None)

    def forward(self, x):
        x = cuda_extension.group_norm_forward(
            x, self.group_norm_weight, self.group_norm_bias,
            self.num_groups, self.eps
        )
        x = cuda_extension.gelu_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.group_norm_forward(
            x, self.group_norm_weight, self.group_norm_bias,
            self.num_groups, self.eps
        )
        yield 'step_001', x
        x = cuda_extension.gelu_forward(x)
        yield 'step_002', x
        yield 'step_003', x