import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, channels, padding_size):
        super().__init__()
        self.channels = channels
        self.padding_size = padding_size

    def forward(self, x):
        x = F.pad(x, self.padding_size, mode="constant", value=0)
        x = torch.mean(x, dim=self.channels)
        return x


batch_size = 1024  # Increased from 16 to increase computational load
channels = 1
height = 512
width = 256
padding_size = (1, 2, 1, 2)  # (left, right, top, bottom)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels, padding_size]