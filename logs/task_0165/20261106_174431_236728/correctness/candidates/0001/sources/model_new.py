import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels_1d,
        out_channels_1d,
        kernel_size_1d,
        stride_1d,
        padding_1d,
        bias_1d,
        in_channels_2d,
        out_channels_2d,
        kernel_size_2d,
        stride_2d,
        padding_2d,
        output_padding_2d,
        bias_2d,
        dim,
    ):
        super().__init__()
        # Keep nn.ConvTranspose modules only for state_dict compatibility; forward uses custom ops.
        self.conv_transpose_1d = nn.ConvTranspose1d(
            in_channels_1d, out_channels_1d, kernel_size_1d, stride=stride_1d, padding=padding_1d, bias=bias_1d
        )
        self.conv_transpose_2d = nn.ConvTranspose2d(
            in_channels_2d,
            out_channels_2d,
            kernel_size_2d,
            stride=stride_2d,
            padding=padding_2d,
            output_padding=output_padding_2d,
            bias=bias_2d,
        )
        self.dim = dim

    def forward(self, x1, x2):
        x1 = cuda_extension.conv_transpose1d_forward(
            x1,
            self.conv_transpose_1d.weight,
            self.conv_transpose_1d.bias,
            self.conv_transpose_1d.stride[0],
            self.conv_transpose_1d.padding[0],
        )
        x2 = cuda_extension.conv_transpose2d_forward(
            x2,
            self.conv_transpose_2d.weight,
            self.conv_transpose_2d.bias,
            self.conv_transpose_2d.stride[0],
            self.conv_transpose_2d.padding[0],
            self.conv_transpose_2d.output_padding[0],
        )
        x1 = cuda_extension.log_softmax_forward(x1, self.dim)
        x2 = cuda_extension.log_softmax_forward(x2, self.dim)
        return x1, x2

    def debug_forward(self, x1, x2):
        y1 = cuda_extension.conv_transpose1d_forward(
            x1,
            self.conv_transpose_1d.weight,
            self.conv_transpose_1d.bias,
            self.conv_transpose_1d.stride[0],
            self.conv_transpose_1d.padding[0],
        )
        yield "step_001", y1
        y2 = cuda_extension.conv_transpose2d_forward(
            x2,
            self.conv_transpose_2d.weight,
            self.conv_transpose_2d.bias,
            self.conv_transpose_2d.stride[0],
            self.conv_transpose_2d.padding[0],
            self.conv_transpose_2d.output_padding[0],
        )
        yield "step_002", y2
        z1 = cuda_extension.log_softmax_forward(y1, self.dim)
        yield "step_003", z1
        z2 = cuda_extension.log_softmax_forward(y2, self.dim)
        yield "step_004", z2
        yield "step_005", (z1, z2)
