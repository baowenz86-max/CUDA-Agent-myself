import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        # Check for NaN and replace with zeros
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = self.prelu(x)
        return x


channels = 64
batch_size = 16
height, width = 512, 256  # Increased width from 128 to 256 to increase computation


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    x[0, 0, 0, 0] = float("nan")  # Add a NaN value for testing
    return [x]


def get_init_inputs():
    return [channels]