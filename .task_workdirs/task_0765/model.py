import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([x1, x2])
        result = torch.bmm(stacked, stacked.transpose(1, 2))
        return result


dim = 8192  # Increased from 512 to increase computation


def get_inputs():
    x1 = torch.randn(dim, 128)
    x2 = torch.randn(dim, 128)
    return [x1, x2]


def get_init_inputs():
    return []