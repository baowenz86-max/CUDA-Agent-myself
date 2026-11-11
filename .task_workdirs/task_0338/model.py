import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.ge(x, self.threshold)


batch_size = 1800
in_channels = 3
height = 512
width = 256
threshold = 0.5


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, threshold]