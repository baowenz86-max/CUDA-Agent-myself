import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__()
        self.instance_norm = nn.InstanceNorm3d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm(x)


batch_size = 2
in_channels = 8
depth = 512
height = 40  # Increased from 30 to 40 to increase computational load
width = 40   # Increased from 30 to 40 to increase computational load


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels]