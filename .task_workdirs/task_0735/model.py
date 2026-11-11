import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

    def forward(self, x):
        x = F.elu(x)
        x = torch.relu(x)
        return x


batch_size = 512  # Increased from 16 to 512 to increase computation
input_channels = 3
width = 512
height = 256


def get_inputs():
    x = torch.randn(batch_size, input_channels, height, width)
    return [x]


def get_init_inputs():
    return [input_channels]