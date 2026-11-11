import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps, momentum, affine, track_running_stats):
        super().__init__()
        self.instance_norm = nn.InstanceNorm3d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        return self.instance_norm(x)


batch_size = 512
num_features = 3
eps = 1e-5
momentum = 0.1
affine = False
track_running_stats = True


def get_inputs():
    return [torch.randn(batch_size, num_features, 48, 48, 48)]


def get_init_inputs():
    return [num_features, eps, momentum, affine, track_running_stats]