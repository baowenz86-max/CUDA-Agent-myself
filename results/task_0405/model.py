import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.abs(x)
        x = F.softmax(x, dim=1)
        return x


batch_size = 128
height, width = 8192, 32  # Increased height to 8192 to increase computation


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []