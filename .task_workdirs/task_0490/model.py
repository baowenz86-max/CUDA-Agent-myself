import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_features, norm_eps):
        super().__init__()
        self.gelu = F.gelu
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=norm_eps)

    def forward(self, x):
        x = self.gelu(x)
        x = self.instance_norm(x)
        return x


num_features = 512
norm_eps = 1e-5


def get_inputs():
    return [torch.randn(16, num_features, 128, 128)]  # Increased spatial dimensions to 128x128


def get_init_inputs():
    return [num_features, norm_eps]