import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = torch.sum(x, dim=self.dim)
        return x


batch_size = 1024  # Increased from 16 to increase computational load
height = 512
width = 256
channels = 3
dim = 1


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [dim]