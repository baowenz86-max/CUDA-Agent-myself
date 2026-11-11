import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels, kernel_size, stride, padding):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.cos(x)
        return x


batch_size = 16
height, width = 3000, 3000  # Increased spatial dimensions to increase computation
input_channels = 3
kernel_size = (3, 5)
stride = 1
padding = (1, 2)


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [input_channels, kernel_size, stride, padding]