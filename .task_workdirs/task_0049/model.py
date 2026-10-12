import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, num_params):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.params = nn.ParameterList([nn.Parameter(torch.randn(out_channels)) for _ in range(num_params)])

    def forward(self, x):
        x = F.conv_transpose1d(
            x,
            self.conv_transpose.weight,
            bias=None,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
        )
        for param in self.params:
            x = x + param.view(1, -1, 1)
        x = torch.round(x)
        return x


batch_size = 128
in_channels = 4
out_channels = 8
length = 8192  # Increased from 512 to 8192 to increase runtime
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
num_params = 3


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, num_params]