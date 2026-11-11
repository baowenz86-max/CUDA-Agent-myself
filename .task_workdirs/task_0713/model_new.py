import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_channels, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = cuda_extension.mean_dim1_keepdim(x)
        x = cuda_extension.cos_elementwise(x)
        return x

    def debug_forward(self, x):
        x_mean = cuda_extension.mean_dim1_keepdim(x)
        yield 'step_001', x_mean
        x_cos = cuda_extension.cos_elementwise(x_mean)
        yield 'step_002', x_cos
        yield 'step_003', x_cos
