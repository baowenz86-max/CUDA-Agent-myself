import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, repeats):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.repeats = repeats

    def forward(self, x):
        x = self.conv_transpose1d(x)
        x = torch.repeat_interleave(x, repeats=self.repeats, dim=2)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
seq_len = 8192  # Increased from 512 to increase computational load
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
repeats = 3


def get_inputs():
    return [torch.randn(batch_size, in_channels, seq_len)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, repeats]