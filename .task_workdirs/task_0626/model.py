import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, out_features, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = F.softplus(x)
        x = torch.sum(x, dim=self.dim, keepdim=True)
        return x


batch_size = 32
in_features = 64
height = 192  # Increased from 128 to 192 to increase runtime
width = 512
dim = 1


def get_inputs():
    x = torch.randn(batch_size, in_features, height, width)
    return [x]


def get_init_inputs():
    return [in_features, None, dim]