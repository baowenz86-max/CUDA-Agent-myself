import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, bias_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = torch.tril(x)
        x = F.elu(x)
        x = x + self.bias
        return x


batch_size = 1024  # Increased from 512 to increase runtime
in_features = 64
height, width = 32, 32
bias_shape = (in_features, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, in_features, height, width)]


def get_init_inputs():
    return [in_features, bias_shape]