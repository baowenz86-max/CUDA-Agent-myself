import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, leakiness):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=leakiness)

    def forward(self, x):
        return self.leaky_relu(x)


batch_size = 1024  # Increased from 512 to increase computation
in_features = 128
height, width = 64, 64
leakiness = 0.01


def get_inputs():
    return [torch.randn(batch_size, in_features, height, width)]


def get_init_inputs():
    return [in_features, leakiness]