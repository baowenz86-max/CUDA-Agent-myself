import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat((x1, x2), dim=self.dim)
        x_amax = torch.amax(x_cat, dim=self.dim)
        return x_amax


# Hyperparameters to control input shape
dim0 = 64
dim1 = 2048  # Increased from 512 to 2048 to increase computation
dim2 = 32
dim3 = 32


def get_inputs():
    x1 = torch.randn(dim0, dim1, dim2, dim3)
    x2 = torch.randn(dim0, dim1, dim2, dim3)
    return [x1, x2]


def get_init_inputs():
    return [1]  # The dimension to concatenate along in 'torch.cat' and the dimension to reduce in 'torch.amax'