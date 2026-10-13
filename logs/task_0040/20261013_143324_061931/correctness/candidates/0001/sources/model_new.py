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
        num_features,
        out_features,
    ):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.linear = nn.Linear(num_features, out_features)

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        x = cuda_extension.channel_add(x, self.bias)
        x = cuda_extension.roll_last_dim(x, 1)
        x = cuda_extension.flatten_to_2d(x)
        x = cuda_extension.linear_forward(x, self.linear.weight, self.linear.bias)
        return x

    def debug_forward(self, x):
        yield 'step_001', cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        x1 = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        x2 = cuda_extension.channel_add(x1, self.bias)
        yield 'step_002', x2
        x3 = cuda_extension.roll_last_dim(x2, 1)
        yield 'step_003', x3
        x4 = cuda_extension.flatten_to_2d(x3)
        yield 'step_004', x4
        x5 = cuda_extension.linear_forward(x4, self.linear.weight, self.linear.bias)
        yield 'step_005', x5
        yield 'step_006', x5
