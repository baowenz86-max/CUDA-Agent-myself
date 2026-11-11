import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, leaky_relu_slope):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

    def forward(self, x):
        x = cuda_extension.custom_mean_dim1(x)
        x = cuda_extension.custom_leaky_relu(x, self.leaky_relu_slope)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_mean_dim1(x)
        yield 'step_001', x
        x = cuda_extension.custom_leaky_relu(x, self.leaky_relu_slope)
        yield 'step_002', x
        yield 'step_003', x

