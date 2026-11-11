import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.softshrink = nn.Softshrink(lambd=lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softshrink(x)


batch_size = 16
channels = 32
height = 1024  # Increased to increase computation
width = 1024   # Increased to increase computation


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [0.5]