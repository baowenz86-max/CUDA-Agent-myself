import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding_size, mean, std):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.normal_layer = torch.nn.Parameter(
            torch.tensor(torch.normal(mean, std, size=(1, 1, 1, 1))), requires_grad=False
        )

    def forward(self, x):
        x = self.reflection_pad(x)
        x = x + self.normal_layer
        return x


batch_size = 16
channels = 3
height = 2048  # Increased from 512
width = 1024   # Increased from 256
padding_size = (3, 3, 3, 3)
mean = 0.0
std = 1.0


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [padding_size, mean, std]