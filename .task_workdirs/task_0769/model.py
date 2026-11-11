import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.softmax(x)
        return x


# Hyperparameters controlling input shape
batch_size = 16
channels = 8
height = 512  # Adjusted to be >=512
width = 1000  # Increased width to increase computation


def get_inputs():
    x = [torch.randn(batch_size, channels, height, width), torch.randn(batch_size, channels, height, width)]
    return [x]


def get_init_inputs():
    return [2, 16]  # num_inputs, num_outputs