import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps, momentum, hard_slope, scale, bias):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum)
        self.hardsigmoid = nn.Hardsigmoid(inplace=False)
        self.scale = scale
        self.bias = bias

    def forward(self, x):
        x = self.instance_norm(x)
        x = self.hardsigmoid(x)
        x = x * self.scale + self.bias
        return x


num_features = 64
eps = 1e-5
momentum = 0.1
hard_slope = 0.2
scale = 0.5
bias = 0.5
batch_size = 512  # Increased batch size to ensure at least one dimension >=512


def get_inputs():
    return [torch.randn(batch_size, num_features, 64, 64)]  # Increased spatial dimensions


def get_init_inputs():
    return [num_features, eps, momentum, hard_slope, scale, bias]