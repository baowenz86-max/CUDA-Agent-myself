import torch
import torch.nn as nn
import math
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, repeats):
        super().__init__()
        self.conv1d_weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.conv1d_bias = nn.Parameter(torch.empty(out_channels))
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = 1
        self.repeats = repeats
        # 初始化参数，参考模型使用默认 Conv1d 初始化
        nn.init.kaiming_uniform_(self.conv1d_weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv1d_bias, -bound, bound)

    def forward(self, x):
        x = cuda_extension.tile_forward(x, self.repeats)
        x = cuda_extension.conv1d_forward(x, self.conv1d_weight, self.conv1d_bias)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x_tiled = cuda_extension.tile_forward(x, self.repeats)
        yield 'step_001', x_tiled
        x_conv = cuda_extension.conv1d_forward(x_tiled, self.conv1d_weight, self.conv1d_bias)
        yield 'step_002', x_conv
        yield 'step_003', x_conv
