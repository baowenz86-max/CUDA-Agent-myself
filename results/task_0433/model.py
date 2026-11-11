import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, kernel_size, dilation, stride):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)

    def forward(self, x):
        x = self.unfold(x)
        x = torch.reshape(x, (x.size(0), -1, kernel_size[0] * kernel_size[1]))
        return x


batch_size = 64  # Increased batch size to increase runtime
in_channels = 3
height = 512
width = 256
kernel_size = (3, 5)
dilation = (2, 1)
stride = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [kernel_size, dilation, stride]