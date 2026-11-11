import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        return torch.log10(torch.abs(attn_output) + 1e-6)


batch_size = 64
seq_length = 128
embed_dim = 512
num_heads = 8


def get_inputs():
    query = torch.randn(batch_size, seq_length, embed_dim)
    key = torch.randn(batch_size, seq_length, embed_dim)
    value = torch.randn(batch_size, seq_length, embed_dim)
    return [query, key, value]


def get_init_inputs():
    return [embed_dim, num_heads, True]