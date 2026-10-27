import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x


batch_size = 1792  # Increased from 16 to increase computation
height = 512
width = 256


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []