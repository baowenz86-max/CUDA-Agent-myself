import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_channels, kernel_size, stride, padding):
        super().__init__()
        self.max = torch.max
        self.unfold = F.unfold
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x_unfolded = self.unfold(x, kernel_size=(2, 2), stride=1, padding=0)
        x_max, _ = self.max(x_unfolded, dim=1)
        x = self.conv(x)
        return x_max, x


batch_size = 20
input_channels = 3
height = 512
width = 256
kernel_size = (3, 5)
stride = 1
padding = (1, 2)


def get_inputs():
    x = torch.randn(batch_size, input_channels, height, width)
    return [x]


def get_init_inputs():
    return [input_channels, kernel_size, stride, padding]