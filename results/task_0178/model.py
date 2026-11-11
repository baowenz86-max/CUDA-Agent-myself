import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        mean_x = torch.mean(x, dim=self.dim, keepdim=True)
        return torch.cat([mean_x, x], dim=self.dim)


def get_inputs():
    # Create a random tensor of size [batch_size, channels, height, width]
    batch_size = 512
    channels = 16
    height = 256  # Increased to increase computation
    width = 256   # Increased to increase computation
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    # The 'dim' argument for torch.mean and torch.cat operations
    return [1]  # Calculate mean along channel dimension