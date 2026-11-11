import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = F.softmax(x, dim=self.dim)
        x = torch.unsqueeze(x, dim=self.dim)
        return x


input_size = 10000
feature_size = 100


def get_inputs():
    return [torch.randn(input_size, feature_size)]


def get_init_inputs():
    return [0]  # The dimension along which softmax will be applied