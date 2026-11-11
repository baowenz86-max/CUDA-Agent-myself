import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        x = cuda_extension.custom_conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias if self.bias else None,
            self.stride,
            self.padding,
            self.output_padding,
        )
        x = cuda_extension.custom_silu(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias if self.bias else None,
            self.stride,
            self.padding,
            self.output_padding,
        )
        yield "step_001", x
        x = cuda_extension.custom_silu(x)
        yield "step_002", x
        yield "step_003", x
