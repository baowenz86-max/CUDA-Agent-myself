import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def forward(self, x):
        x = F.elu(x)
        x = torch.mean(x)
        return x


batch_size = 1024  # Increased batch size to increase computation
height = 512
width = 256
in_channels = 3


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return []