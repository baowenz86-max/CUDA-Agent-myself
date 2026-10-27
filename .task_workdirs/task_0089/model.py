import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x + self.scale_factor
        return x


# Define input dimensions with at least one dimension >= 512
dim1 = 64
dim2 = 64
dim3 = 512  # Adjusted to 512 to meet the requirement
dim4 = 128


def get_inputs():
    return [torch.randn(dim1, dim2, dim3, dim4)]


def get_init_inputs():
    return [2.0]  # An arbitrary scale factor