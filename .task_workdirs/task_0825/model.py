import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        index_select_dim,
        index_select_indices,
        conv_weight_shape,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.index_select_dim = index_select_dim
        self.index_select_indices = nn.Parameter(index_select_indices, requires_grad=False)
        self.conv_weight = nn.Parameter(torch.randn(conv_weight_shape), requires_grad=True)

    def forward(self, x):
        x = torch.index_select(x, self.index_select_dim, self.index_select_indices)
        x = F.conv_transpose1d(
            x,
            self.conv_weight,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
        )
        return x


batch_size = 2048
in_channels = 3
out_channels = 128
height = 2048  # Increased to 2048 to provide larger input
kernel_size = 4
stride = 2
padding = 1
output_padding = 0
index_select_dim = 2
index_select_indices = torch.arange(height, dtype=torch.long)  # Use full height indices
conv_weight_shape = (in_channels, out_channels, kernel_size)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        index_select_dim,
        index_select_indices,
        conv_weight_shape,
    ]