import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.elu(x)
        return x


batch_size = 5000  # Increased batch size to increase computation
width = 512
height = 128


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    # No initial parameters needed for this operation
    return []