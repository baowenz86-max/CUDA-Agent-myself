import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return cuda_extension.maxpool2d_forward(x, self.maxpool.kernel_size, self.maxpool.stride, self.maxpool.padding, self.maxpool.dilation, self.maxpool.ceil_mode)

    def debug_forward(self, x):
        out = self.forward(x)
        yield 'step_001', out
