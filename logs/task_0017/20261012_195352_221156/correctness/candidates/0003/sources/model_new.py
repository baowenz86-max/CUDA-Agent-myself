import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # Preserve state_dict keys exactly
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Store convolution attributes for custom op call
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        # Custom conv transpose1d: input, weight, bias, stride, padding, output_padding
        x = cuda_extension.conv_transpose1d_custom(
            x,
            self.conv_transpose.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        x = cuda_extension.min_scalar_custom(x, 1.0)
        x = cuda_extension.erf_custom(x)
        return x

    def debug_forward(self, x):
        # Same computational path, but yield intermediate tensors with checkpoint ids
        x = cuda_extension.conv_transpose1d_custom(
            x,
            self.conv_transpose.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        yield 'step_001', x
        x = cuda_extension.min_scalar_custom(x, 1.0)
        yield 'step_002', x
        x = cuda_extension.erf_custom(x)
        yield 'step_003', x
        yield 'step_004', x
