import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sum(x, dim=1)
        x = torch.log_softmax(x, dim=0)
        return x


batch_size = 2048  # Increased batch size to increase computation
width = 512
height = 256


def get_inputs():
    x = torch.randn(batch_size, 3, height, width)
    return [x]


def get_init_inputs():
    return []  # No parameters needed for this particular forward function