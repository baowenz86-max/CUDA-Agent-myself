import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.abs(x)
        x = F.log_softmax(x, dim=1)
        return x


batch_size = 512


def get_inputs():
    # Increased feature dimension (dim=1) to increase computation load for log_softmax
    return [torch.randn(batch_size, 256, 64, 64)]


def get_init_inputs():
    return []