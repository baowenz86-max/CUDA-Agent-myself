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
        conv1d_out_channels,
        conv1d_kernel_size,
        conv1d_stride,
        conv1d_padding,
        dim,
    ):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.conv1d = nn.Conv1d(
            out_channels, conv1d_out_channels, conv1d_kernel_size, stride=conv1d_stride, padding=conv1d_padding
        )
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        x = cuda_extension.conv1d_forward(
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            self.conv1d.stride[0],
            self.conv1d.padding[0],
        )
        x = cuda_extension.sum_dim_forward(x, self.dim)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        yield "step_001", x
        x = cuda_extension.conv1d_forward(
            x,
            self.conv1d.weight,
            self.conv1d.bias,
            self.conv1d.stride[0],
            self.conv1d.padding[0],
        )
        yield "step_002", x
        x = cuda_extension.sum_dim_forward(x, self.dim)
        yield "step_003", x
        yield "step_004", x
