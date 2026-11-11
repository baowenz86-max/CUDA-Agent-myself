import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        x = torch.prod(x, dim=1, keepdim=True)
        x = torch.clamp(x, min=0, max=1)
        return x


batch_size = 16
height = 512
width = 256
channels = 512  # Increased channels to add computational load


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return []