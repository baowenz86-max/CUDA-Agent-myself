import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.sigmoid(x)
        return x


batch_size = 512
channels = 32
height, width = 72, 72


def get_inputs():
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    return [x1, x2]


def get_init_inputs():
    return [32, 64]