import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = torch.triu(x, diagonal=1)
        x = torch.abs(x)
        return x


# Hyperparameters determining input shape
batch_size = 64
channels = 3
height = 1024
width = 1024


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    # The input size for the model is assumed to be of the form [batch_size, channels, height, width]
    return [224]