import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        another_in_channels,
        conv1d_weight_shape,
        conv1d_bias_shape,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.conv1d_weight = nn.Parameter(torch.randn(conv1d_weight_shape))
        self.conv1d_bias = nn.Parameter(torch.randn(conv1d_bias_shape)) if conv1d_bias_shape is not None else None
        self.another_in_channels = another_in_channels

    def forward(self, x, y):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose.weight,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
        )
        x = cuda_extension.add_bias_forward(x, self.bias)
        y = cuda_extension.conv1d_forward(
            y,
            self.conv1d_weight,
            self.conv1d_bias,
            stride=1,
            padding=0,
        )
        return (x, y)

    def debug_forward(self, *inputs):
        x, y = inputs
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose.weight,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
        )
        yield ("step_001", x)
        x = cuda_extension.add_bias_forward(x, self.bias)
        yield ("step_002", x)
        y = cuda_extension.conv1d_forward(
            y,
            self.conv1d_weight,
            self.conv1d_bias,
            stride=1,
            padding=0,
        )
        yield ("step_003", y)
        yield ("step_004", (x, y))
