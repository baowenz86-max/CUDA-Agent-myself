import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        # Keep the same structure as Model.__init__ for state_dict compatibility
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.triu_forward(x)
        x = cuda_extension.conv2d_forward(
            x, self.conv.weight, self.conv.bias,
            stride=1, padding=0, dilation=1, groups=1
        )
        x = cuda_extension.add_bias_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.triu_forward(x)
        yield 'step_001', x1
        x2 = cuda_extension.conv2d_forward(
            x1, self.conv.weight, self.conv.bias,
            stride=1, padding=0, dilation=1, groups=1
        )
        yield 'step_002', x2
        x3 = cuda_extension.add_bias_forward(x2, self.bias)
        yield 'step_003', x3
        yield 'step_004', x3
