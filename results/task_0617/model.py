import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.glu = nn.GLU(dim=1)
        self.equal_check = torch.equal

    def forward(self, x):
        x = self.glu(x)
        return x


def get_inputs():
    batch_size = 1024  # Increased to increase computation
    seq_length = 1024  # Increased to increase computation
    input_dim = 256    # Increased to increase computation
    x = torch.randn(batch_size, seq_length, input_dim)
    return [x]


def get_init_inputs():
    return [256, 256]  # Fixed to valid integer dimensions for model init