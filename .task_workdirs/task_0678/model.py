import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        x1 = torch.flip(x1, [self.dim])
        x2 = torch.flip(x2, [self.dim])
        return torch.cat((x1, x2), dim=self.dim)


# Hyperparameters for input dimensions
dim1 = 16
dim2 = 32
dim3 = 2048  # Increased to 2048 to meet runtime requirement
dim4 = 64


def get_inputs():
    x1 = torch.randn(dim1, dim2, dim3, dim4)
    x2 = torch.randn(dim1, dim2, dim3, dim4)
    return [x1, x2]


def get_init_inputs():
    return [2]  # dimension for concatenation along.