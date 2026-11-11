import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, num_groups, eps):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps)

    def forward(self, x):
        x = self.group_norm(x)
        x = torch.softmax(x, dim=1)
        return x


batch_size = 512
num_channels = 3
height, width = 256, 256  # Increased spatial dimensions to 256x256
num_groups = 3
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width)]


def get_init_inputs():
    return [num_channels, num_groups, eps]