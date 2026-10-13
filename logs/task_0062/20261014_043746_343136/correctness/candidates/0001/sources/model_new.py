import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
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
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        x = cuda_extension.sigmoid_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        yield 'step_001', x
        x = cuda_extension.sigmoid_forward(x)
        yield 'step_002', x
        yield 'step_003', x
