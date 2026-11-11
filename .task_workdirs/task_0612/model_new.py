import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        x = cuda_extension.custom_sum_dim(x, self.in_channels)
        x = cuda_extension.custom_expm1(x)
        return x

    def debug_forward(self, x):
        sum_result = cuda_extension.custom_sum_dim(x, self.in_channels)
        yield ('step_001', sum_result)
        expm1_result = cuda_extension.custom_expm1(sum_result)
        yield ('step_002', expm1_result)
        yield ('step_003', expm1_result)
