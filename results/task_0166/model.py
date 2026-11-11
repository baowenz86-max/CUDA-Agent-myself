import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = torch.log_softmax(x, dim=1)
        return x


batch_size, height, width = 128, 8192, 32


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []