import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_size, num_params):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pad = nn.ReflectionPad2d(padding_size)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(out_channels)) for _ in range(num_params)])

    def forward(self, x):
        x = cuda_extension.reflection_pad2d(x, padding=1)
        x = cuda_extension.conv2d(x, self.conv.weight, self.conv.bias, stride=1, padding=0)
        return x

    def debug_forward(self, x):
        x = cuda_extension.reflection_pad2d(x, padding=1)
        yield 'step_001', x
        x = cuda_extension.conv2d(x, self.conv.weight, self.conv.bias, stride=1, padding=0)
        yield 'step_002', x
        yield 'step_003', x
