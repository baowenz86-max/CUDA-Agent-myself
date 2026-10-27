import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = cuda_extension.sqrt_forward(x)
        x = cuda_extension.round_forward(x)
        x = cuda_extension.conv_transpose2d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0],
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.sqrt_forward(x)
        yield 'step_001', x
        x = cuda_extension.round_forward(x)
        yield 'step_002', x
        x = cuda_extension.conv_transpose2d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0],
        )
        yield 'step_003', x
        yield 'step_004', x
