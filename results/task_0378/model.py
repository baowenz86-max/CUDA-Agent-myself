import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x):
        return self.batch_norm(x)


batch_size = 32  # Increased from 16 to 32 to increase runtime
num_features = 3
channels = 64
width = 256
height = 512
eps = 1e-05
momentum = 0.1
affine = True
track_running_stats = True


def get_inputs():
    x = torch.randn(batch_size, num_features, height, width)
    return [x]


def get_init_inputs():
    return [num_features, eps, momentum, affine, track_running_stats]