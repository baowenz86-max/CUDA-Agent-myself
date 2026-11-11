import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, min_val):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.min_val = min_val

    def forward(self, x):
        x = self.batch_norm(x)
        x = torch.clamp(x, min=self.min_val)
        return x


def get_inputs():
    batch_size = 16
    num_features = 64
    height = 512
    width = 256  # Increased from 128 to 256 to increase computation
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    num_features = 64
    min_val = 0.0
    return [num_features, min_val]