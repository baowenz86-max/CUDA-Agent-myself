import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim, index):
        super().__init__()
        self.dim = dim
        self.index = index
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.index_select(x, self.dim, self.index)
        x = torch.round(x)
        x = self.maxpool(x)
        return x


batch_size = 512
channels = 16
height = 256
width = 256


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [1, torch.tensor([0, 3, 6, 9])]