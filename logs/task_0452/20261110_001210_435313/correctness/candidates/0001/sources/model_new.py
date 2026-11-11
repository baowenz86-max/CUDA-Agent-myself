import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, bias_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # x must be CUDA float32 tensor, shape [N,C,H,W]
        x = cuda_extension.custom_sigmoid(x)
        x = cuda_extension.custom_relu(x)
        x = cuda_extension.custom_add_bias(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_sigmoid(x)
        yield 'step_001', x
        x = cuda_extension.custom_relu(x)
        yield 'step_002', x
        x = cuda_extension.custom_add_bias(x, self.bias)
        yield 'step_003', x
        yield 'step_004', x
