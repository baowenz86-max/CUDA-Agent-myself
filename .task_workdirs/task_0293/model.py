import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, eps):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_channels, eps=eps)

    def forward(self, x):
        x = self.instance_norm(x)
        x = torch.log_softmax(x, dim=1)
        return x


batch_size = 128
num_channels = 3
height, width = 512, 640  # Increased width to 640 to increase computation
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels, eps]