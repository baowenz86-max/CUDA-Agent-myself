import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = nn.Softmax()(x)
        return x


batch_size = 1
channels = 512  # Increased from 3 to increase computation
height = 512
width = 224


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []