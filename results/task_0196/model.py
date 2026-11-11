import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.gelu(x)
        x = torch.tanh(x)
        return x


batch_size = 512  # Increased from 64 to increase runtime
width = 512
height = 1024


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []