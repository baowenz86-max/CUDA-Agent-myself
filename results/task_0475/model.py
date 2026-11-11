import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.einsum = "ijk->ijk"

    def forward(self, x):
        x = torch.einsum(self.einsum, x)
        x = F.elu(x)
        return x


batch_size = 512
height, width = 1024, 1024


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []