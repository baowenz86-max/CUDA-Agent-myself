import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=0)
        x = torch.view_as_complex(x)
        return x


batch_size = 4096  # Increased from 128 to increase computation time
real_dim = 512
complex_dim = 256


def get_inputs():
    return [torch.randn(batch_size, real_dim, complex_dim, 2)]


def get_init_inputs():
    return []