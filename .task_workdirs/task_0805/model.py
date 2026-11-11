import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, num_channels, stack_dim):
        super().__init__()
        self.stack_dim = stack_dim
        self.relu = F.relu
        self.input_size = input_size
        self.num_channels = num_channels
        self.stack = nn.Parameter(
            torch.stack([torch.randn(self.input_size) for _ in range(self.num_channels)], dim=self.stack_dim)
        )

    def forward(self, x):
        x = torch.stack([self.relu(x) for _ in range(self.num_channels)], dim=self.stack_dim)
        return x


# Hyperparameters for input shape
batch_size = 512
channels = 3
height = 256
width = 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [3, 4, 1]