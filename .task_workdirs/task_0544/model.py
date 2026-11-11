import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_features, eps):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, eps)

    def forward(self, x):
        x = F.relu(x)
        x = self.batch_norm(x)
        return x


num_features = 256
eps = 1e-5
batch_size = 1024  # Increased from 16 to 1024 to increase runtime
width = 512  # Maintained to keep one dimension >=512


def get_inputs():
    x = torch.randn(batch_size, num_features, width)
    return [x]


def get_init_inputs():
    return [num_features, eps]