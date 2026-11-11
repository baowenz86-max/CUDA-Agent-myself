import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.softsign = nn.Softsign()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign(x)


batch_size = 16
channels = 3
height = 2048
width = 2048


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)  # Increased height and width to 2048
    return [x]


def get_init_inputs():
    return []  # No parameters needed for the Softsign layer