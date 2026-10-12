import torch
import torch.nn as nn


class Model(nn.Module):
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
        x = self.conv_transpose(x)
        x = self.layer_norm(x)
        x = torch.addcmul(x, v, w, value=1.0)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
length = 6400  # Increased from 256 to 6400 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
layer_norm_shape = (out_channels, (length - 1) * stride - 2 * padding + kernel_size + output_padding)
weight_shape = layer_norm_shape
bias_shape = layer_norm_shape


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    v = torch.randn(batch_size, out_channels, (length - 1) * stride - 2 * padding + kernel_size + output_padding)
    w = torch.randn(batch_size, out_channels, (length - 1) * stride - 2 * padding + kernel_size + output_padding)
    return [x, v, w]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        layer_norm_shape,
        weight_shape,
        bias_shape,
    ]