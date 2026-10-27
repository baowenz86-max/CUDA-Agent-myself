import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = torch.triu(x)
        x = self.conv(x)
        x = x + self.bias
        return x

    def debug_forward(self, x):
        x = torch.triu(x)
        yield 'step_001', x
        x = self.conv(x)
        yield 'step_002', x
        x = x + self.bias
        yield 'step_003', x
        yield 'step_004', x
