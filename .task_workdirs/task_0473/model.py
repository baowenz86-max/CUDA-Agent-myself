import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardtanh(x)


batch_size = 1280  # Increased batch size to increase runtime
channels = 3
height = 512
width = 256


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return []  # No parameters needed for this model initialization