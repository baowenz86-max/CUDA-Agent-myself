import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, reduction_dim: tuple):
        super().__init__()
        self.reduction_dim = reduction_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sum(x, self.reduction_dim)
        x = torch.round(x)
        return x


batch_size = 512
channels = 3
height = 768
width = 768


def get_inputs():
    # Example input tensor of shape [batch_size, channels, height, width]
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    # Specify the dimensions along which to reduce the tensor
    return [(2, 3)]