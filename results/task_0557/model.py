import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mish = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mish(x)


batch_size = 2048  # Increased from 512 to increase computation
in_channels = 3
height = 256
width = 256


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return []