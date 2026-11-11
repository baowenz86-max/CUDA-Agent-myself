import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
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
        self.log_softmax = nn.LogSoftmax(dim=dim)

    def forward(self, x1, x2):
        x1 = F.conv_transpose1d(
            x1,
            self.conv_transpose_1d.weight,
            bias=self.conv_transpose_1d.bias,
            stride=self.conv_transpose_1d.stride,
            padding=self.conv_transpose_1d.padding,
        )
        x2 = self.conv_transpose_2d(x2)
        x1 = self.log_softmax(x1)
        x2 = self.log_softmax(x2)
        return x1, x2


batch_size = 512
in_channels_1d = 3
out_channels_1d = 16
kernel_size_1d = 3
stride_1d = 2
padding_1d = 1
bias_1d = True
in_channels_2d = 3
out_channels_2d = 16
kernel_size_2d = 3
stride_2d = 2
padding_2d = 1
output_padding_2d = 1
bias_2d = True
dim = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels_1d, 64), torch.randn(batch_size, in_channels_2d, 32, 32)]


def get_init_inputs():
    return [
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
    ]