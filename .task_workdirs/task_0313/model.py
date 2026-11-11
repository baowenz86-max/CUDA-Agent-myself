import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batchnorm2d(x)
        x = torch.prod(x, dim=1, keepdim=True)
        return x


batch_size = 16
num_features = 3
height = 1024
width = 512


def get_inputs():
    x = torch.randn(batch_size, num_features, height, width)
    return [x]


def get_init_inputs():
    return [num_features]