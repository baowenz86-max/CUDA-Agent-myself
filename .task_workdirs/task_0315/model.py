import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tanh_layer = nn.Tanh()
        self.leaky_relu_layer = nn.LeakyReLU()

    def forward(self, x):
        x = self.tanh_layer(x)
        x = self.leaky_relu_layer(x)
        return x


def get_inputs():
    # Increased spatial dimensions to increase computation
    batch_size = 512
    channels = 32
    height = 128  # Increased from 32
    width = 128   # Increased from 32
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    # No specific initialization arguments needed for this model
    return []