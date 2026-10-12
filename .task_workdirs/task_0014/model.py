import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias,
        clamp_min,
        clamp_max,
        mask_val,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.mask_val = mask_val

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        mask = x != self.mask_val
        x = torch.masked_select(x, mask)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
length = 4096  # Increased from 512 to 4096 to increase runtime
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias = False
clamp_min = -1.0
clamp_max = 1.0
mask_val = -1


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias,
        clamp_min,
        clamp_max,
        mask_val,
    ]