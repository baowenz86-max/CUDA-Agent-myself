import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps):
        super(Model, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=eps)

    def forward(self, x):
        x = self.instance_norm(x)
        x = x + 1.0  # 'add' operation
        return x


batch_size = 16
channels = 32
height = 512
width = 512  # Increased width from 64 to 512 to increase computation


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [32, 1e-5]