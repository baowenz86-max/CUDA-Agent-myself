import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_groups, num_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.conv_transpose1d = nn.ConvTranspose1d(
            num_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        x = cuda_extension.custom_group_norm(x, self.num_groups)
        x = cuda_extension.custom_conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_group_norm(x, self.num_groups)
        yield ('step_001', x)
        x = cuda_extension.custom_conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        yield ('step_002', x)
        yield ('step_003', x)
