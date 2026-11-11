import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = torch.ops.cuda_extension.conv3d_forward(x, self.conv3d.weight, self.conv3d.bias, self.conv3d.stride, self.conv3d.padding)
        x = torch.ops.cuda_extension.gelu_forward(x)
        x = torch.ops.cuda_extension.bias_add_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        x = torch.ops.cuda_extension.conv3d_forward(x, self.conv3d.weight, self.conv3d.bias, self.conv3d.stride, self.conv3d.padding)
        yield 'step_001', x
        x = torch.ops.cuda_extension.gelu_forward(x)
        yield 'step_002', x
        x = torch.ops.cuda_extension.bias_add_forward(x, self.bias)
        yield 'step_003', x
        yield 'step_004', x
