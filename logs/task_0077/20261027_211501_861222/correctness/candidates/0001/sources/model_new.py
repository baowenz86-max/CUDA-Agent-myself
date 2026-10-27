import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(bias_shape)

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.conv_transpose.stride[0], self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0], self.conv_transpose.groups,
            self.conv_transpose.dilation[0]
        )
        x = cuda_extension.layer_norm_forward(
            x, self.norm.weight, self.norm.bias, self.norm.eps
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.conv_transpose.stride[0], self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0], self.conv_transpose.groups,
            self.conv_transpose.dilation[0]
        )
        yield 'step_001', x
        x = cuda_extension.layer_norm_forward(
            x, self.norm.weight, self.norm.bias, self.norm.eps
        )
        yield 'step_002', x
        yield 'step_003', x
