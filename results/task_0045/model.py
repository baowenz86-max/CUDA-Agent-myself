import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sigmoid(x)
        return x


batch_size = 1024
seq_len = 1024
feature_dim = 512


def get_inputs():
    return [torch.randn(batch_size, seq_len, feature_dim)]


def get_init_inputs():
    return []