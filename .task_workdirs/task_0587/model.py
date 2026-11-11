import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, num_groups, epsilon):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=epsilon)

    def forward(self, x):
        x = torch.where(torch.isinf(x), torch.tensor(float("nan"), device=x.device), x)
        x = self.group_norm(x)
        return x


batch_size = 512
num_channels = 3
num_groups = 3
height, width = 256, 256  # Increased from 32x32 to 256x256
epsilon = 1e-5


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width).uniform_(-1000, 1000)]


def get_init_inputs():
    return [num_channels, num_groups, epsilon]