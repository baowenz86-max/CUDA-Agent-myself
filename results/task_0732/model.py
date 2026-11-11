import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, mid_dim):
        super().__init__()
        self.mid_dim = mid_dim

    def forward(self, x):
        # Apply acos to each channel independently
        x = torch.stack([torch.acos(xi) for xi in torch.unbind(x, dim=1)], dim=1)
        # Concatenate along the channel dimension
        x = torch.cat(torch.unbind(x, dim=1), dim=2)
        return x


batch_size = 16
in_channels = 64  # Increased from 3 to 64 to increase computational load
height = 512
width = 256
mid_dim = 64


def get_inputs():
    # Generate a tensor with values restricted to [-1, 1]
    x = torch.randn(batch_size, in_channels, height, width).clamp(-1, 1)
    return [x]


def get_init_inputs():
    return [in_channels, mid_dim]