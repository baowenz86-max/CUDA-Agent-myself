import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        layer_norm_shape,
        weight_shape,
        bias_shape,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.layer_norm = nn.LayerNorm(layer_norm_shape)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, v, w):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
        )
        x = self.layer_norm(x)
        x = cuda_extension.addcmul_forward(x, v, w, 1.0)
        return x

    def debug_forward(self, x, v, w):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
        )
        yield 'step_001', x
        x = self.layer_norm(x)
        yield 'step_002', x
        x = cuda_extension.addcmul_forward(x, v, w, 1.0)
        yield 'step_003', x
        yield 'step_004', x
