import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.log_softmax(x, dim=1)
        x = torch.sum(x, dim=1)
        return x


batch_size = 256
seq_len = 512
features = 1024


def get_inputs():
    x = torch.randn(batch_size, seq_len, features)  # Input shape (256, 512, 1024)
    return [x]


def get_init_inputs():
    return []