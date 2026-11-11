import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, histc_bins, group, eps):
        super().__init__()
        self.sin = torch.sin
        self.histc = torch.histc
        self.group_norm = F.group_norm
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.histc_bins = histc_bins
        self.group = group
        self.eps = eps

    def forward(self, x):
        x = self.sin(x)
        x = self.conv(x)
        x = self.group_norm(x, self.group, eps=self.eps)
        x = self.histc(x, bins=self.histc_bins)
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 64, 64  # Increased spatial dimensions from 32x32 to 64x64
kernel_size = 3
stride = 1
padding = 1
dilation = 1
bias = True
histc_bins = 10
group = 2
eps = 1e-10


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, bias, histc_bins, group, eps]