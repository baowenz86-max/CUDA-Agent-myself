import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.relu6 = nn.ReLU6()
        self.out_features = out_features

    def forward(self, x):
        x = self.relu6(x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = x.expand(-1, self.out_features, -1, -1)
        return x


batch_size = 16
in_channels = 3
out_channels = 64
height = 2560  # Increased to increase computation time
width = 2560   # Increased to increase computation time


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels]