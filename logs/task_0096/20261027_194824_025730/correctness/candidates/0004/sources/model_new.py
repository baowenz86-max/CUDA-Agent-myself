import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dim):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        x = cuda_extension.softmax(x, self.dim)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        yield 'step_001', x
        x = cuda_extension.softmax(x, self.dim)
        yield 'step_002', x
        yield 'step_003', x
