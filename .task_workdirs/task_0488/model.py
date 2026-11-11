import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.softmax(x, dim=1)
        x = F.kl_div(x, x, reduction="batchmean")
        return x


batch_size = 32
height = 2048  # Increased from 64 to 2048 to increase computation
width = 512


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []