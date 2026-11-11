import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = F.linear(x, self.linear.weight, self.bias)
        return x


batch_size = 512
seq_len = 1000
in_features = 128
out_features = 256
bias_shape = (out_features,)


def get_inputs():
    return [torch.randn(batch_size, seq_len, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]