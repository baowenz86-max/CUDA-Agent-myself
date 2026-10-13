import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = nn.Hardtanh(min_val=-1.0, max_val=1.0)(x)
        return x


batch_size = 512
height = 768
width = 768


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []