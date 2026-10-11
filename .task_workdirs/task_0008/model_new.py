import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        return cuda_extension.fused_pad_rotate_conv_maskinf_addbias(
            x, self.conv.weight, self.conv.bias, self.bias,
            self.padding, self.kernel_size
        )
