import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channels, num_groups, eps):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.eps = eps
        self.num_groups = num_groups
        self.num_channels = num_channels

    def forward(self, x):
        x = torch.view_as_real(x)
        x = self.log_softmax(x)
        x = F.group_norm(x, num_groups=self.num_groups, eps=self.eps)
        return x


batch_size = 512
num_channels = 8
height, width = 128, 128  # Increased spatial dimensions to increase computation
num_groups = 4
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, num_channels, height, width, dtype=torch.complex64)]


def get_init_inputs():
    return [num_channels, num_groups, eps]