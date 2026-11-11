import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps):
        super(Model, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps)

    def forward(self, x):
        x = self.sigmoid(x)
        x = self.instance_norm(x)
        return x


batch_size = 16
channels = 3
height = 2048  # Increased from 512
width = 512    # Increased from 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [channels, 1e-5]