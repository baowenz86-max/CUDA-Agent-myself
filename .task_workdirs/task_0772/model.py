import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels_concat, kernel_size, stride, padding, bias_shape_1, bias_shape_2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels_concat, kernel_size, stride=stride, padding=padding)
        self.bias1 = nn.Parameter(torch.randn(bias_shape_1))
        self.conv2 = nn.Conv2d(in_channels, out_channels_concat, kernel_size, stride=stride, padding=padding)
        self.bias2 = nn.Parameter(torch.randn(bias_shape_2))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x1 + self.bias1
        x2 = self.conv2(x)
        x2 = x2 + self.bias2
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(x)
        return x


batch_size = 512
in_channels = 3
out_channels_concat = 16
height, width = 128, 128  # Increased from 32x32 to 128x128
kernel_size = 3
stride = 2
padding = 1
bias_shape_1 = (out_channels_concat, 1, 1)
bias_shape_2 = (out_channels_concat, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels_concat, kernel_size, stride, padding, bias_shape_1, bias_shape_2]