import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.maxpool1d = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool1d(x)
        x = self.tanh(x)
        return x


batch_size = 128
in_channels = 32
out_channels = 64
kernel_size = 3
width = 16384  # Increased to increase computation load


def get_inputs():
    x = torch.randn(batch_size, in_channels, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]