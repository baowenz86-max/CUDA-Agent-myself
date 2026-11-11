import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_channels, eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x, _ = torch.max(x, dim=1, keepdim=True)  # max over channels
        x = self.layer_norm(x)
        return x


batch_size = 512
num_channels = 64
eps = 1e-5


def get_inputs():
    x = torch.randn(
        batch_size, num_channels, 1024, 64
    )  # Increased height from 64 to 1024 to increase runtime
    return [x]


def get_init_inputs():
    return [num_channels, eps]