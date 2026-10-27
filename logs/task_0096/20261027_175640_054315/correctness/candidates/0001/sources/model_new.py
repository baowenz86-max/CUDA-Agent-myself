import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, dim):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_ext(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        x = cuda_extension.softmax_dim1_ext(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose1d_ext(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        yield 'step_001', x
        x = cuda_extension.softmax_dim1_ext(x)
        yield 'step_002', x
        yield 'step_003', x
