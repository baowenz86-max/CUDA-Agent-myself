import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_size, seq_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)
        self.batch_size = batch_size
        self.seq_length = seq_length

    def forward(self, x1, x2):
        q = self.query_linear(x1)
        k = self.key_linear(x2)
        v = self.value_linear(x2)
        attn = F.scaled_dot_product_attention(q, k, v)
        cat = torch.cat((x1, attn), dim=-1)
        return cat


embed_dim = 512
num_heads = 8
batch_size = 16
seq_length = 1024  # Increased from 128 to 1024 to increase computation


def get_inputs():
    x1 = torch.randn(batch_size, seq_length, embed_dim)
    x2 = torch.randn(batch_size, seq_length, embed_dim)
    return [x1, x2]


def get_init_inputs():
    return [embed_dim, num_heads, batch_size, seq_length]