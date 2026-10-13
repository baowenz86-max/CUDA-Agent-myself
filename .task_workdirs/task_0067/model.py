import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = F.elu(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = x + self.mean
        return x


def get_inputs():
    return [torch.randn(16, 3, 2048, 2048)]  # Increased spatial dimensions to 2048x2048


def get_init_inputs():
    return []