import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = torch.tril(x, diagonal=0)  # Applies the Lower-triangle of a matrix
        return self.instance_norm(x)


batch_size = 16
channels = 3
height = 1024  # Increased from 512 to increase computation
width = 1024   # Increased from 32 to increase computation


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels]