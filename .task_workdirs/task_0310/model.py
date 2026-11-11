import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        x = self.instance_norm(x)
        x = torch.nn.functional.selu(x)
        return x


def get_inputs():
    num_features = 64
    batch_size = 1024  # Increased batch size to increase runtime
    height, width = 32, 32
    x = torch.randn(batch_size, num_features, height, width)
    return [x]


def get_init_inputs():
    num_features = 64
    return [num_features]