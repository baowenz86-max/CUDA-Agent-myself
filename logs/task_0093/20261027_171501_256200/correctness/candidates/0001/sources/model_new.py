import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.custom_conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride,
            self.conv_transpose1d.padding,
            self.conv_transpose1d.output_padding,
        )
        x = cuda_extension.custom_mul(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride,
            self.conv_transpose1d.padding,
            self.conv_transpose1d.output_padding,
        )
        yield 'step_001', x
        x = cuda_extension.custom_mul(x, self.bias)
        yield 'step_002', x
        yield 'step_003', x
