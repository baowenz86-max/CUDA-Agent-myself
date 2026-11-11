import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x1, x2):
        x1 = self.linear(x1)
        x2 = self.conv2d(x2)
        return x1, x2


batch_size = 128
in_features = 768
out_features = 512
in_channels = 3
out_channels = 64
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    x1 = torch.randn(batch_size, in_features)
    x2 = torch.randn(batch_size, in_channels, 128, 128)  # Increased spatial dimensions from 32x32 to 128x128
    return [x1, x2]


def get_init_inputs():
    return [in_features, out_features, in_channels, out_channels, kernel_size, stride, padding]