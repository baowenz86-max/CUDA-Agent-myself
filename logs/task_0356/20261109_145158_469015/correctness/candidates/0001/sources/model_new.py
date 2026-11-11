import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, num_channels, num_groups):
        super().__init__()
        # Preserve state_dict keys exactly as original BatchNorm2d and GroupNorm
        self.batch_norm = nn.Module()
        self.batch_norm.weight = nn.Parameter(torch.ones(num_features))
        self.batch_norm.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('batch_norm.running_mean', torch.zeros(num_features))
        self.register_buffer('batch_norm.running_var', torch.ones(num_features))
        self.register_buffer('batch_norm.num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.batch_norm.eps = 1e-5
        self.batch_norm.momentum = 0.1

        self.group_norm = nn.Module()
        self.group_norm.weight = nn.Parameter(torch.ones(num_channels))
        self.group_norm.bias = nn.Parameter(torch.zeros(num_channels))
        self.group_norm.eps = 1e-5
        self.group_norm.num_groups = num_groups
        self.num_channels = num_channels

    def forward(self, x):
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        x = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.eps,
            self.group_norm.num_groups,
            self.num_channels
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps
        )
        yield 'step_001', x
        x = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.eps,
            self.group_norm.num_groups,
            self.num_channels
        )
        yield 'step_002', x
        yield 'step_003', x
