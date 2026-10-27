import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, padding):
        super().__init__()
        self.in_channels = in_channels
        self.padding = nn.ReflectionPad2d(padding)

    def forward(self, x):
        x = cuda_extension.log_softmax(x, dim=1)
        x = cuda_extension.softmax(x, dim=1)
        x = cuda_extension.reflection_pad2d(x, padding=2)
        return x

    def debug_forward(self, x):
        x = cuda_extension.log_softmax(x, dim=1)
        yield ('step_001', x)
        x = cuda_extension.softmax(x, dim=1)
        yield ('step_002', x)
        x = cuda_extension.reflection_pad2d(x, padding=2)
        yield ('step_003', x)
        yield ('step_004', x)
