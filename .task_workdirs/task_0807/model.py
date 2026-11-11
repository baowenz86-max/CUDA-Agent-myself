import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.out_features = out_features

    def forward(self, x):
        x = self.relu(x)
        x = torch.cos(x) + 0.1 * self.out_features
        return x


batch_size = 16
in_features = 64
out_features = 1024
height = 512
width = 512  # Increased from 256 to 512


def get_inputs():
    x = torch.randn(batch_size, in_features, height, width)
    return [x]


def get_init_inputs():
    return [in_features, out_features]