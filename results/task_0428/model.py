import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, divisor: int = 1):
        super().__init__()
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.abs(x)
        x = torch.div(x, self.divisor)
        return x


batch_size = 2048  # Increased from 16 to 2048 to increase computation
width = 512
height = 256
divisor = 2


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return [divisor]