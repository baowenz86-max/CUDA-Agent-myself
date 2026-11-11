import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = self.instance_norm(x)
        return x


batch_size = 16
num_features = 3
height = 2048
width = 1024


def get_inputs():
    x = torch.randn(batch_size, num_features, height, width)
    return [x]


def get_init_inputs():
    return [num_features]