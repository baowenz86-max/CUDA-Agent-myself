import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, normalized_shape, eps):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        x = torch.abs(x)
        x = F.layer_norm(x, self.normalized_shape, eps=self.eps)
        return x


batch_size = 512
channels = 128
height, width = 64, 64  # Increased from 28x28 to 64x64
normalized_shape = (channels, height, width)
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [normalized_shape, eps]