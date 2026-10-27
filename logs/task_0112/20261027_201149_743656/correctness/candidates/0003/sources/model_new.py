import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        weight = self.conv_transpose.weight + self.bias
        x = cuda_extension.custom_conv_transpose1d(
            x,
            weight,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0]
        )
        x = cuda_extension.custom_softmax_dim1(x)
        return x

    def debug_forward(self, *inputs):
        x, = inputs
        weight = self.conv_transpose.weight + self.bias
        x = cuda_extension.custom_conv_transpose1d(
            x,
            weight,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0]
        )
        yield 'step_001', x
        x = cuda_extension.custom_softmax_dim1(x)
        yield 'step_002', x
        yield 'step_003', x
