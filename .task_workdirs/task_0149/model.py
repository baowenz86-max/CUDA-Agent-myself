import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim, vdim, batch_first):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, kdim=kdim, vdim=vdim, batch_first=batch_first)

    def forward(self, x, query, key, value):
        x = torch.view_as_complex(x)
        output, _ = self.multihead_attn(query, key, value)
        return output


batch_size = 16
seq_length = 2048  # Increased from 512 to increase computational load
embed_dim = 256
num_heads = 8
kdim = 256
vdim = 256
batch_first = True


def get_inputs():
    x = torch.randn(batch_size, seq_length, embed_dim // 2, 2)  # For view_as_complex
    query = torch.randn(batch_size, seq_length, embed_dim)
    key = torch.randn(batch_size, seq_length, kdim)
    value = torch.randn(batch_size, seq_length, vdim)
    return [x, query, key, value]


def get_init_inputs():
    return [embed_dim, num_heads, kdim, vdim, batch_first]