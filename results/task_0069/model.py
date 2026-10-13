import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sigmoid(x)
        x = F.softmax(x, dim=1)
        return x


batch_size = 512
channels = 10
height = 200  # Increased from 20 to 200
width = 200   # Increased from 20 to 200
features = 4


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []  # No initialization parameters required