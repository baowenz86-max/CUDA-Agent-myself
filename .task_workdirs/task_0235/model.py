import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps, affine, num_groups):
        super(Model, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, eps=eps, affine=affine)

    def forward(self, x):
        return torch.einsum("bchw->bhwc", torch.sum(self.group_norm(x), dim=1, keepdim=True))


batch_size = 512  # Increased to ensure at least one dimension >=512
num_features = 3
eps = 1e-5
affine = True
num_groups = 3


def get_inputs():
    x = torch.randn(batch_size, num_features, 256, 256)  # Increased spatial dimensions to 256x256
    return [x]


def get_init_inputs():
    return [num_features, eps, affine, num_groups]