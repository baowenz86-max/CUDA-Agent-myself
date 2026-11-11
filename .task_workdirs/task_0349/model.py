import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=0)
        x = nn.Hardswish()(x)
        return x


def get_inputs():
    # Create a batch of random tensors for input
    batch_size = 16
    width = 4096  # Increased from 256 to 4096 to increase computation
    height = 4096  # Increased from 512 to 4096 to increase computation
    channels = 3
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    # Since there are no initialization parameters for the model in this case,
    # we return an empty list.
    return []