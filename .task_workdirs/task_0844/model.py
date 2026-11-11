import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x, attn_mask=None):
        return torch.mean(self.multihead_attn(x, x, x, attn_mask=attn_mask)[0], dim=1)


embed_dim = 512
num_heads = 8
seq_len = 1024  # Increased sequence length
batch_size = 32  # Added batch dimension


def get_inputs():
    x = torch.rand((seq_len, batch_size, embed_dim))  # Added batch dimension
    attn_mask = torch.rand((seq_len, seq_len))
    return [x, attn_mask]


def get_init_inputs():
    return [embed_dim, num_heads]