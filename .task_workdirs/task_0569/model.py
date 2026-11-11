import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x)
        x = self.flatten(x)
        return x


batch_size = 1280  # Increased from 16 to increase computation
channels = 3
height = 512
width = 256


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return []