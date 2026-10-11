import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv2d(x)
        x = cuda_extension.triu_forward(x, diagonal=0)
        x = cuda_extension.bitwise_xor_uint8_forward(x, self.bias)
        return x
