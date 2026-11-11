import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_features, eps):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, eps=eps)

    def forward(self, x):
        x = F.hardsigmoid(x)
        x = self.norm(x)
        return x


batch_size = 16
num_features = 3
height = 1024  # Increased to increase computation
width = 1024   # Increased to increase computation
eps = 1e-05


def get_inputs():
    x = torch.randn(batch_size, num_features, height, width)
    return [x]


def get_init_inputs():
    return [num_features, eps]