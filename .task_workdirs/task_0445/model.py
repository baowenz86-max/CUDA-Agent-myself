import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps, affine):
        super().__init__()
        self.sync_batch_norm = nn.SyncBatchNorm(num_features, eps, affine)

    def forward(self, x):
        x = torch.abs(x)
        x = self.sync_batch_norm(x)
        return x


num_features = 128
eps = 1e-5
affine = True
batch_size = 512  # Increased to meet the dimension requirement
height, width = 64, 64  # Increased spatial dimensions to increase runtime


def get_inputs():
    x = torch.randn(batch_size, num_features, height, width)
    return [x]


def get_init_inputs():
    return [num_features, eps, affine]