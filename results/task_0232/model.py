import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x):
        x = self.instance_norm(x)
        x = x * 2
        return x


batch_size = 17
num_features = 64
height = 512
width = 256


def get_inputs():
    x = torch.randn(batch_size, num_features, height, width)
    return [x]


def get_init_inputs():
    return [num_features]