import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.conv2d_forward(x, self.conv2d.weight, self.conv2d.bias)
        x = cuda_extension.triu_forward(x)
        x = cuda_extension.xor_forward(x, self.bias)
        x = cuda_extension.to_float_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv2d_forward(x, self.conv2d.weight, self.conv2d.bias)
        yield "step_001", x
        x = cuda_extension.triu_forward(x)
        yield "step_002", x
        x = cuda_extension.xor_forward(x, self.bias)
        yield "step_003", x
        x = cuda_extension.to_float_forward(x)
        yield "step_004", x
