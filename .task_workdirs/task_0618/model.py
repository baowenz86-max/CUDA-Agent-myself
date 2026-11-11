import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = torch.softmax(x, dim=1)
        return x


batch_size = 16
in_channels = 64  # Increased from 3 to increase computation
width = 512
height = 256


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels]