import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.instancenorm3d = nn.InstanceNorm3d(50)

    def forward(self, x):
        x = self.softplus(x)
        x = self.instancenorm3d(x)
        return x


batch_size = 16
channels = 50
depth = 512
height = 20
width = 20


def get_inputs():
    return [torch.randn(batch_size, channels, depth, height, width)]


def get_init_inputs():
    return []