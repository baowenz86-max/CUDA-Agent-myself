import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        x = torch.cos(x)
        x = self.identity(x)
        return x


batch_size = 512
height, width = 1024, 1024  # Increased from 64x64 to 1024x1024


def get_inputs():
    return [torch.randn(batch_size, 1, height, width)]


def get_init_inputs():
    return []  # No parameters needed for this model