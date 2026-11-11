import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.sync_bn = nn.SyncBatchNorm(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # Apply torch.max
        x = self.sync_bn(x)
        return x, x_max


num_features = 32
batch_size = 512
height, width = 64, 64  # Increased spatial dimensions to increase computation


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features]