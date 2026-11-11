import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(in_features, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x)
        x = self.instance_norm(x)
        return x


in_features = 64
height, width = 512, 288  # Increased width from 256 to 288 to increase runtime


def get_inputs():
    return [torch.randn(16, in_features, height, width)]


def get_init_inputs():
    return [in_features]