import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_shape, eps=eps)

    def forward(self, x):
        x = torch.abs(x)
        x = self.layer_norm(x)
        return x


batch_size = 512
in_channels = 3
height, width = 300, 300  # Increased from 32x32 to 300x300
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    input_shape = (in_channels, height, width)
    return [input_shape, eps]