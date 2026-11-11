import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, num_groups, eps):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.num_channels = num_channels

    def forward(self, x):
        x = cuda_extension.view_as_real_complex64(x)
        x = cuda_extension.log_softmax_dim_last(x)
        x = cuda_extension.group_norm_fp32(x, self.num_groups, self.eps)
        return x

    def debug_forward(self, x):
        x = cuda_extension.view_as_real_complex64(x)
        yield 'step_001', x
        x = cuda_extension.log_softmax_dim_last(x)
        yield 'step_002', x
        x = cuda_extension.group_norm_fp32(x, self.num_groups, self.eps)
        yield 'step_003', x
        yield 'step_004', x
