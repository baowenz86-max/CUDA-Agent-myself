import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, intermediate_channels, groups, p
    ):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.conv1d = nn.Conv1d(out_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        self.p = p

    def forward(self, x):
        x = F.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            bias=self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride,
            padding=self.conv_transpose1d.padding,
            output_padding=self.conv_transpose1d.output_padding,
        )
        x = F.normalize(x, p=self.p, dim=1)
        x = self.conv1d(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
intermediate_channels = 8
groups = 1
p = 2.0
length = 8192  # Increased from 512 to 8192 to increase computational load


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, intermediate_channels, groups, p]