import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_features, bias_shape, hardsigmoid_slope):
        super().__init__()
        self.batch_norm2d = nn.BatchNorm2d(num_features)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardsigmoid_slope = hardsigmoid_slope

    def forward(self, x):
        x = self.batch_norm2d(x)
        x = F.hardsigmoid(x, self.hardsigmoid_slope)
        x = x + self.bias
        return x


batch_size = 512
num_features = 3
bias_shape = (num_features, 1, 1)
hardsigmoid_slope = 0.2


def get_inputs():
    return [torch.randn(batch_size, num_features, 128, 128)]  # Increased spatial dimensions


def get_init_inputs():
    return [num_features, bias_shape, hardsigmoid_slope]