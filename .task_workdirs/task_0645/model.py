import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sum(x, dim=1)
        x = self.identity(x)
        return x


batch_size = 16
width = 512
height = 256
channels = 2048  # Increased channel dimension to increase computation


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return []