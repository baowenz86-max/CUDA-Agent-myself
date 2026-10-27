import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.hardswish = nn.Hardswish()
        self.softplus = nn.Softplus()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = cuda_extension.hardswish_forward(x)
        x = cuda_extension.softplus_forward(x)
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride[0],
            padding=self.conv_transpose1d.padding[0],
            output_padding=self.conv_transpose1d.output_padding[0]
        )
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.hardswish_forward(x)
        yield 'step_001', x1
        x2 = cuda_extension.softplus_forward(x1)
        yield 'step_002', x2
        x3 = cuda_extension.conv_transpose1d_forward(
            x2,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride[0],
            padding=self.conv_transpose1d.padding[0],
            output_padding=self.conv_transpose1d.output_padding[0]
        )
        yield 'step_003', x3
        yield 'step_004', x3
