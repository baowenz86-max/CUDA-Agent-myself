import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = torch.clamp(x, min=self.min_value, max=self.max_value)
        x = x - self.min_value
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
height, width = 1024, 1024  # Increased spatial dimensions to increase computation
min_value = -0.5
max_value = 0.5


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return [min_value, max_value]