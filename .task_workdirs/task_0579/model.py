import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pad_width, pad_mode, value
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.pad_width = pad_width
        self.pad_mode = pad_mode
        self.value = value

    def forward(self, x):
        x = F.conv_transpose1d(
            x,
            self.conv_transpose.weight,
            bias=self.conv_transpose.bias,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
        )
        x = F.pad(x, self.pad_width, mode=self.pad_mode, value=self.value)
        return x


batch_size = 512
in_channels = 3
out_channels = 16
length = 4096  # Increased from 32 to 4096 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pad_width = (2, 2)
pad_mode = "constant"
value = 0.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pad_width, pad_mode, value]