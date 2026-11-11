import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.circular_pad = nn.CircularPad1d(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circular_pad(x)


batch_size = 128
channels = 64
length = 26000  # Increased to increase runtime
padding = 16


def get_inputs():
    x = torch.randn(batch_size, channels, length)
    return [x]


def get_init_inputs():
    return [padding]