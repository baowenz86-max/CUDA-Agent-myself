import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.selu(x)
        x = torch.cumsum(x, dim=0)
        return x


seq_len = 20000  # Increased from 512 to increase computation
feature_dim = 5


def get_inputs():
    return [torch.randn(seq_len, feature_dim)]  # Larger input tensor

def get_init_inputs():
    return []  # No initialization parameters needed