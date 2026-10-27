import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, histc_bins, histc_min, histc_max
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.histc_bins = histc_bins
        self.histc_min = histc_min
        self.histc_max = histc_max

    def forward(self, x):
        x = self.conv_transpose(x)
        hist = torch.histc(x, bins=self.histc_bins, min=self.histc_min, max=self.histc_max)
        return x, hist


batch_size = 128
in_channels = 3
out_channels = 16
length = 4096  # Increased from 512 to 4096 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
histc_bins = 100
histc_min = 0
histc_max = 100


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, histc_bins, histc_min, histc_max]