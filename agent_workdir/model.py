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
<<<<<<< HEAD
channels = 10
depth = 15
height = 15
width = 15


def get_inputs():
    return [torch.randn(batch_size, channels, depth, height, width)]


def get_init_inputs():
    return []
=======
in_channels = 32
num_groups = 4
num_features = in_channels
height, width = 128, 128  # Increased from 32x32 to 128x128 to increase computation


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, num_groups, num_features]
>>>>>>> b8d3180 (Update implementation with new approach)
