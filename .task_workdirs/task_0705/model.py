import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, indices = torch.max(x, dim=self.dim)
        x = self.bn(indices.float())
        return x


batch_size = 256
num_features = 128
dim = 1
length = 50000  # Increased to make max operation more computationally intensive


def get_inputs():
    x = torch.randn(batch_size, length, num_features)
    return [x]


def get_init_inputs():
    return [num_features, dim]