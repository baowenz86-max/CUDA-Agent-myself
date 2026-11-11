import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = torch.exp(x)
        return x


batch_size = 16
height, width = 2560, 160  # Increased dimensions to increase computation


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []