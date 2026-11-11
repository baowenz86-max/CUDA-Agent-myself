import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # Check for NaNs and remove them
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x, _ = self.attention(x, x, x)
        return x


embed_dim = 64
num_heads = 4


def get_inputs():
    # Assuming attention requires 3D input with shape: (seq_length, batch_size, embed_dim)
    batch_size, seq_length = 128, 1024  # Increased seq_length to 1024
    x = torch.randn(seq_length, batch_size, embed_dim)
    return [x]


def get_init_inputs():
    return [embed_dim, num_heads]