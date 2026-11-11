import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = cuda_extension.unfold_forward(x, self.unfold.kernel_size, self.unfold.dilation, self.unfold.padding, self.unfold.stride)
        x = cuda_extension.cat_forward(x, x, dim=1)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.unfold_forward(x, self.unfold.kernel_size, self.unfold.dilation, self.unfold.padding, self.unfold.stride)
        yield 'step_001', x
        x = cuda_extension.cat_forward(x, x, dim=1)
        yield 'step_002', x
        yield 'step_003', x
