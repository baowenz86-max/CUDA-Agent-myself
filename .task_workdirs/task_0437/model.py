import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = F.relu(x + 3) / 6
        x = self.hardswish(x)
        return x


batch_size = 1024  # Increased from 512 to increase computation
channels = 32
height = 64
width = 64


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)  # example tensor
    return [x]


def get_init_inputs():
    # No specific parameters to return for this case
    return []