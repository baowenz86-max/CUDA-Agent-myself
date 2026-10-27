import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, clamp_min, clamp_max, power_factor):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.power_factor = power_factor

    def forward(self, x):
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        x = torch.pow(x, self.power_factor)
        return x


clamp_min = -0.5
clamp_max = 0.5
power_factor = 2.0
batch_size = 1024  # Increased to increase computation


def get_inputs():
    return [torch.randn(batch_size, 128, 128, 16)]  # Larger spatial dimensions


def get_init_inputs():
    return [clamp_min, clamp_max, power_factor]