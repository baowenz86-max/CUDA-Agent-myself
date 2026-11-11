import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(input, target)


batch_size = 16
channels = 3
height = 2048
width = 2048


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)
    return [x, y]


def get_init_inputs():
    return []