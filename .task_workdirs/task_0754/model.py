import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, num_features):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features

    def forward(self, x):
        x1 = torch.cat((x, x), dim=1)
        x2 = torch.maximum(x1, x1)
        return x2


batch_size = 16
in_channels = 3
width = 2048
height = 2048  # Increased spatial dimensions to increase computation


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, in_channels * 2]