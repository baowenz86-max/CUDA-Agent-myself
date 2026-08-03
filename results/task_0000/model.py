import torch
from torch import digamma, max
from torch.nn import BatchNorm3d, Parameter


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = BatchNorm3d(10)
        self.parameter = Parameter(torch.randn(10))

    def forward(self, x):
        x = self.batch_norm(x)
        x = digamma(x)
        x = max(x)
        x = x + self.parameter
        return x


batch_size = 512
channels = 10
depth = 15
height = 15
width = 15


def get_inputs():

    x = torch.empty(
        batch_size,
        channels,
        depth,
        height,
        width
    ).uniform_(0.1, 5.0)

    return [x]


def get_init_inputs():
    return []