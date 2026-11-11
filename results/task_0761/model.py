import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = F.softmax(x, dim=dim)
        x = torch.squeeze(x)
        x = x + self.epsilon  # to prevent numerical instability in softmax
        return x


batch_size, seq_len, hidden_dim = 32, 1024, 512
dim = 1


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_dim)]


def get_init_inputs():
    return [dim]