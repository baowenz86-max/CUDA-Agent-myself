import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        conv_out = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose1d.weight, self.conv_transpose1d.bias
        )
        mul_out = cuda_extension.mul_bias_forward(conv_out, self.bias)
        return mul_out

    def debug_forward(self, x):
        conv_out = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose1d.weight, self.conv_transpose1d.bias
        )
        yield 'step_001', conv_out
        mul_out = cuda_extension.mul_bias_forward(conv_out, self.bias)
        yield 'step_002', mul_out
        yield 'step_003', mul_out
