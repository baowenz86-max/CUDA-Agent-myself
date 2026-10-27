import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_size, reduction_factor, bias):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        self.pooling = nn.AdaptiveAvgPool1d(pooling_size)
        self.reduction_factor = reduction_factor

    def forward(self, x):
        x = cuda_extension.custom_conv1d(x, self.conv1d.weight, self.conv1d.bias, self.conv1d.stride, self.conv1d.padding, self.conv1d.dilation, self.conv1d.groups)
        x = cuda_extension.custom_adaptive_avg_pool1d(x, self.pooling.output_size)
        x = cuda_extension.custom_slice(x, dim=2, start=0, end=self.reduction_factor)
        x = cuda_extension.custom_std(x, dim=2)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_conv1d(x, self.conv1d.weight, self.conv1d.bias, self.conv1d.stride, self.conv1d.padding, self.conv1d.dilation, self.conv1d.groups)
        yield "step_001", x
        x = cuda_extension.custom_adaptive_avg_pool1d(x, self.pooling.output_size)
        yield "step_002", x
        x = cuda_extension.custom_slice(x, dim=2, start=0, end=self.reduction_factor)
        yield "step_003", x
        x = cuda_extension.custom_std(x, dim=2)
        yield "step_004", x
        yield "step_005", x
