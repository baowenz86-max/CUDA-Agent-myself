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
        min_val,
        max_val,
        bias_bitwise_val,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.min_val = min_val
        self.max_val = max_val
        self.bias_bitwise_val = bias_bitwise_val

    def forward(self, x):
        x = F.conv_transpose1d(
            x,
            self.conv_transpose.weight,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
        )
        x = torch.clip(x, min=self.min_val, max=self.max_val)
        x = torch.bitwise_xor(x.type(torch.int64), torch.tensor(self.bias_bitwise_val, dtype=torch.int64))
        return x


batch_size = 512
in_channels = 8
out_channels = 16
length = 2048  # Increased from 32 to 2048
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
min_val = -1.0
max_val = 1.0
bias_bitwise_val = 1


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, min_val, max_val, bias_bitwise_val]