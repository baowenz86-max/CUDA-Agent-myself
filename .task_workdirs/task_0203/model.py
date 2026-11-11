import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.gelu(x)
        x = torch.trunc(x)
        return x


batch_size = 16
width = 2560
height = 5120  # Increased dimensions to increase runtime


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    # No additional initialization parameters required for this case
    return []