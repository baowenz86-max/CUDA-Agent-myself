import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.erf(x)
        x = torch.erf(x)
        x = torch.median(x, dim=0)
        return x


def get_inputs():
    # Increased input size to [16384, 1024] to increase runtime
    return [torch.randn(16384, 1024)]


def get_init_inputs():
    # No parameters needed for the forward function in this scenario
    return []