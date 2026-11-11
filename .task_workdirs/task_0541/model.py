import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, num_features, eps, momentum):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.bn(x)
        x = self.gelu(x)
        return x


batch_size = 2560  # Increased from 512 to increase runtime
num_features = 3
eps = 1e-5
momentum = 0.1
height, width = 32, 32


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features, num_features, eps, momentum]