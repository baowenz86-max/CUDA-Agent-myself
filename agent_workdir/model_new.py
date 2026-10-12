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
        x = cuda_extension.log10_lt_forward(x, threshold=0.5)
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride[0],
            padding=self.conv_transpose1d.padding[0],
            output_padding=self.conv_transpose1d.output_padding[0],
            dilation=self.conv_transpose1d.dilation[0],
            groups=self.conv_transpose1d.groups,
        )
        x = cuda_extension.bias_add_forward(x, self.bias)
        return x
