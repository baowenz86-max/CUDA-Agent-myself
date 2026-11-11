import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool(x)


def get_inputs():
    batch_size = 16
    channels = 3
    depth = 512
    height = 128  # Increased from 32 to 128
    width = 128   # Increased from 32 to 128
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]


def get_init_inputs():
    return []