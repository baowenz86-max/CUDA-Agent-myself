import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias, seq_modules):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.seq = nn.Sequential(*seq_modules)

    def forward(self, x):
        x = self.conv_transpose1d(x)
        x = self.seq(x)
        return x


batch_size = 64
in_channels = 16
out_channels = 32
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias = True
seq_length = 5120  # Increased from 512 to increase runtime
seq_modules = [
    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2, stride=2),
]


def get_inputs():
    return [torch.randn(batch_size, in_channels, seq_length)]  # Use updated seq_length


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias, seq_modules]