import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, num_groups, eps):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps)

    def forward(self, x):
        x = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            num_groups=self.group_norm.num_groups,
            eps=self.group_norm.eps,
        )
        x = cuda_extension.softmax_channel_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            num_groups=self.group_norm.num_groups,
            eps=self.group_norm.eps,
        )
        yield 'step_001', x
        x = cuda_extension.softmax_channel_forward(x)
        yield 'step_002', x
        yield 'step_003', x
