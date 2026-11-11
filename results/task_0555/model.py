import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, normalize_shape, dim):
        super().__init__()

    def forward(self, x):
        # Apply torch.max
        x, _ = torch.max(x, dim=1, keepdim=True)
        # Apply F.normalize
        x = F.normalize(x, p=2.0, dim=dim)
        return x


batch_size = 128
height, width = 1024, 256  # Increased spatial dimensions
dim = 2  # Change normalization dimension to height


def get_inputs():
    return [torch.randn(batch_size, 32, height, width)]  # Increased channels to 32


def get_init_inputs():
    return [None, dim]  # As we don't need to initialize any parameters