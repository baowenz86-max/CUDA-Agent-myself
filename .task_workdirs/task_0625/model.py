import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sum(x, dim=1, keepdim=True)  # sum along the channel dimension
        x = torch.clip(x, min=0, max=1)  # clip the output between 0 and 1
        return x


batch_size = 512
in_channels = 3
height = 512
width = 512


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    # No parameters needed for init as there are no learnable parameters
    return []