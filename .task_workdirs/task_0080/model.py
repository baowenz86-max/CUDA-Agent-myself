import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters, init)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.masked_select(x, mask)
        return self.prelu(x)


def get_inputs():
    batch_size = 16
    height = 1024  # Increased from 512
    width = 1024   # Increased from 512
    channels = 3
    x = torch.randn(batch_size, channels, height, width)
    mask = torch.randn(channels, height, width) > 0
    return [x, mask]


def get_init_inputs():
    return []