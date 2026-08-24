import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, num_groups, num_features):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features

    def forward(self, x):
        x = torch.ceil(x)
        x = torch.transpose(x, 1, 2)
        x = F.group_norm(x, self.num_groups)
        return x


batch_size = 512
in_channels = 32
num_groups = 4
num_features = in_channels
height, width = 128, 128  # Increased from 32x32 to 128x128 to increase computation


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, num_groups, num_features]