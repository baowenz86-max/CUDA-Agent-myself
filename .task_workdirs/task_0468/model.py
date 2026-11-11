import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hard_sigmoid = nn.Hardtanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hard_sigmoid(x)


batch_size = 64
in_channels = 3
height = 2048
width = 2048


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return []