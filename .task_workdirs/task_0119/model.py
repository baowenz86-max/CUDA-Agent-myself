import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_features, num_classes, eps, momentum, affine):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        x = self.batch_norm(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        return x


batch_size = 64
num_features = 1024
num_classes = 10
eps = 1e-05
momentum = 0.1
affine = True


def get_inputs():
    return [torch.randn(batch_size, num_features, 64, 64)]


def get_init_inputs():
    return [num_features, num_classes, eps, momentum, affine]