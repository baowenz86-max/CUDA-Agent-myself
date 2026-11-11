import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps, momentum, affine):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        return self.instance_norm(x)


num_features = 64
eps = 1e-5
momentum = 0.1
affine = True
batch_size = 2048  # Added batch size to increase input volume
seq_length = 512


def get_inputs():
    x = torch.randn(batch_size, num_features, seq_length)  # 3D input for InstanceNorm1d (N, C, L)
    return [x]


def get_init_inputs():
    return [num_features, eps, momentum, affine]