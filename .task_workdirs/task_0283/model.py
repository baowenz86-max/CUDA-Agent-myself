import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.elu(x)
        x = torch.tanh(x)
        return x


batch_size = 16
features = 32
height, width = 1024, 1024  # Increased spatial dimensions to increase computation


def get_inputs():
    x = torch.randn(batch_size, features, height, width)
    return [x]


def get_init_inputs():
    return []