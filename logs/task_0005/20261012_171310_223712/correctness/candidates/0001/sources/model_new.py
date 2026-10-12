import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.custom_log10(x)
        x = cuda_extension.custom_lt(x, 0.5)
        x = cuda_extension.custom_conv_transpose1d(
            x.float(),
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
            self.conv_transpose1d.dilation[0],
            self.conv_transpose1d.groups
        )
        x = cuda_extension.custom_add_bias(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_log10(x)
        yield "step_001", x
        x = cuda_extension.custom_lt(x, 0.5)
        yield "step_002", x
        x = cuda_extension.custom_conv_transpose1d(
            x.float(),
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
            self.conv_transpose1d.dilation[0],
            self.conv_transpose1d.groups
        )
        yield "step_003", x
        x = cuda_extension.custom_add_bias(x, self.bias)
        yield "step_004", x
        yield "step_005", x
        return
