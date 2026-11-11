import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, threshold):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.threshold = threshold

    def forward(self, x):
        x = self.conv(x)
        condition = x > self.threshold
        x = torch.where(condition, x, torch.zeros_like(x))
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 128, 128  # Increased from 32x32 to 128x128
kernel_size = 3
threshold = 0.5


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, threshold]