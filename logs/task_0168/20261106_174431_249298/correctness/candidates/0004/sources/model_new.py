import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = cuda_extension.conv_transpose_op(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
        )
        x = cuda_extension.add_bias_op(x, self.bias)
        x = cuda_extension.softmax_channel_op(x)
        x = cuda_extension.scale_op(x, self.scaling_factor)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose_op(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
        )
        yield "step_001", x
        x = cuda_extension.add_bias_op(x, self.bias)
        yield "step_002", x
        x = cuda_extension.softmax_channel_op(x)
        yield "step_003", x
        x = cuda_extension.scale_op(x, self.scaling_factor)
        yield "step_004", x
        yield "step_005", x
