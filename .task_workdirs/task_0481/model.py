import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, linear_out_features, bias=True):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, bias=bias)
        self.glu = nn.GLU(dim=-1)
        self.linear = nn.Linear(embed_dim, linear_out_features)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        x = torch.cat((attn_output, attn_output), dim=-1)
        x = self.glu(x)
        x = self.linear(x)
        return x


batch_size = 16
seq_len = 1024  # Increased from 512 to increase computation
embed_dim = 256
num_heads = 8
linear_out_features = 128


def get_inputs():
    # Changed shape order to (seq_len, batch_size, embed_dim) for MultiheadAttention compatibility
    query = torch.randn(seq_len, batch_size, embed_dim)
    key = torch.randn(seq_len, batch_size, embed_dim)
    value = torch.randn(seq_len, batch_size, embed_dim)
    return [query, key, value]


def get_init_inputs():
    return [embed_dim, num_heads, linear_out_features]