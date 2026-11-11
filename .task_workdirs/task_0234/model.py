import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batchnorm(x)
        x = torch.abs(x)
        return x


batch_size = 512
num_features = 16


def get_inputs():
    x = torch.randn(batch_size, num_features, 128, 128)  # Increased spatial dimensions from 32x32 to 128x128
    return [x]


def get_init_inputs():
    return [num_features]