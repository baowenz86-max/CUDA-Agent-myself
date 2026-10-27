import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding_constant: int, padding_reflection: tuple):
        super().__init__()
        self.padding_constant = nn.ConstantPad2d(padding_constant, 0)
        self.padding_reflection = nn.ReflectionPad2d(padding_reflection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding_constant(x)
        x = self.padding_reflection(x)
        return x


batch_size = 16
height = 2048  # Increased from 512 to increase computation
width = 1024   # Increased from 256 to increase computation
channels = 3
padding_constant = (1, 2, 3, 4)
padding_reflection = (1, 2, 3, 4)


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [padding_constant, padding_reflection]