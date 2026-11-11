import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, norm_size, alpha, beta, k):
        super().__init__()
        self.norm_size = norm_size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        x = F.local_response_norm(x, self.norm_size, self.alpha, self.beta, self.k)
        return x


batch_size = 48
in_channels = 3
height = 512
width = 256
norm_size = 5
alpha = 1e-4
beta = 0.75
k = 2


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, norm_size, alpha, beta, k]