import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = torch.neg(x)
        x = self.pool(x)
        x = torch.addcmul(self.bias, x, x)
        return x


in_features = 64
out_features = 16
batch_size = 128
time_steps = 15360  # Increased from 512 to increase runtime


def get_inputs():
    return [torch.randn(batch_size, in_features, time_steps)]


def get_init_inputs():
    return [in_features, out_features, (out_features,)]