import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride=stride)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.reflection_pad.padding[0], self.reflection_pad.padding[2])
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias, self.conv.stride[0], self.conv.padding[0], self.conv.dilation[0], self.conv.groups)
        x = cuda_extension.glu_forward(x, 1)
        x = cuda_extension.min_dim_forward(x, 1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(x, self.reflection_pad.padding[0], self.reflection_pad.padding[2])
        yield "step_001", x
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias, self.conv.stride[0], self.conv.padding[0], self.conv.dilation[0], self.conv.groups)
        yield "step_002", x
        x = cuda_extension.glu_forward(x, 1)
        yield "step_003", x
        x = cuda_extension.min_dim_forward(x, 1)
        yield "step_004", x
        yield "step_005", x
