import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, batch_first, stack_dim):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=batch_first)
        self.stack_dim = stack_dim

    def forward(self, x1, x2, x3, key_padding_mask):
        x1, _ = self.multihead_attn(x1, x2, x3, key_padding_mask=key_padding_mask)
        x4 = torch.multiply(x1, x2)
        return torch.stack((x3, x4), dim=self.stack_dim)


batch_size = 128
seq_len = 64  # Increased from 32 to increase computation
embed_dim = 512
num_heads = 8
stack_dim = 1


def get_inputs():
    x1 = torch.randn(batch_size, seq_len, embed_dim)
    x2 = torch.randn(batch_size, seq_len, embed_dim)
    x3 = torch.randn(batch_size, seq_len, embed_dim)
    key_padding_mask = torch.rand(batch_size, seq_len) > 0.5
    return [x1, x2, x3, key_padding_mask]


def get_init_inputs():
    return [embed_dim, num_heads, True, True, stack_dim]