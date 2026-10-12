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
        x = cuda_extension.conv_transpose1d_custom(
            x,
            self.conv_transpose.weight,
            self.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
        )
        x = cuda_extension.min_scalar_custom(x, 1.0)
        x = cuda_extension.erf_custom(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose1d_custom(
            x,
            self.conv_transpose.weight,
            self.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
        )
        yield "step_001", x
        x = cuda_extension.min_scalar_custom(x, 1.0)
        yield "step_002", x
        x = cuda_extension.erf_custom(x)
        yield "step_003", x
        yield "step_004", x
