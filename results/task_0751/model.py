import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.linear(x)
        x = torch.minimum(x, self.bias)
        return x


batch_size = 8192  # Increased from 1024 to increase computation
in_features = 2048
out_features = 1024
bias_shape = (out_features,)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]