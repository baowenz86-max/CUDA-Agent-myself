import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = F.layer_norm(x, self.normalized_shape)
        return x


batch_size = 1024  # Increased from 128 to increase runtime
features = 1024
width = 32
height = 32
normalized_shape = (1, width, height)


def get_inputs():
    return [torch.randn(batch_size, features, height, width)]


def get_init_inputs():
    return [normalized_shape]