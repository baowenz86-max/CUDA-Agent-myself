import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, query, key, value):
        x, _ = self.attention(query, key, value)
        return torch.matmul(x, value.transpose(1, 2))


embed_dim = 128
num_heads = 4
batch_size = 32  # Increased from 16 to increase computation
seq_len = 1024  # Increased from 512 to increase computation


def get_inputs():
    query = key = value = torch.randn(batch_size, seq_len, embed_dim)
    return [query, key, value]


def get_init_inputs():
    return [embed_dim, num_heads]