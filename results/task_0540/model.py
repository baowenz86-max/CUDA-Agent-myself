import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x = self.batch_norm(x)
        x = torch.clamp(x, min=0.1)
        return x


num_features = 16
batch_size = 512
height, width = 64, 64  # Increased spatial dimensions to 64x64


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features]