import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.elu(x)  # Apply Exponential Linear Unit (ELU)
        x = torch.digamma(x)  # Apply Digamma function
        x = torch.triu(x)  # Apply Triangular Upright
        return x


batch_size = 2048
feature_dim = 4096


def get_inputs():
    return [torch.randn(batch_size, feature_dim)]  # Increased input size to boost runtime


def get_init_inputs():
    return []  # No initialization parameters needed for this case