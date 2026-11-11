import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_channels, kernel_size, stride, padding):
        super().__init__()
        self.hardtanh = F.hardtanh
        self.avgpool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.hardtanh(x)
        x = self.avgpool(x)
        return x


input_channels = 3
kernel_size = (2, 2)
stride = 1
padding = (0, 0)
batch_size = 16
height = 2048
width = 2048


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [input_channels, kernel_size, stride, padding]