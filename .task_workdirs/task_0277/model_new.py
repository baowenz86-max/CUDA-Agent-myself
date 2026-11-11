import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # 保存卷积参数，用于自定义算子调用
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        x = cuda_extension.selu_forward(x)
        x = cuda_extension.conv2d_forward(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.selu_forward(x)
        yield 'step_001', x
        x = cuda_extension.conv2d_forward(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        yield 'step_002', x
        yield 'step_003', x
