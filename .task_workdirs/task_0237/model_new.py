import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, padding_size, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.padding_size = padding_size
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.reflection_pad2d_forward(x, self.padding_size[0], self.padding_size[1], self.padding_size[2], self.padding_size[3])
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.add_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.reflection_pad2d_forward(x, self.padding_size[0], self.padding_size[1], self.padding_size[2], self.padding_size[3])
        yield 'step_001', x
        x = cuda_extension.relu_forward(x)
        yield 'step_002', x
        x = cuda_extension.add_forward(x, self.bias)
        yield 'step_003', x
        yield 'step_004', x
