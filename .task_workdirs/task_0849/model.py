import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, sequence_length, repeat_interleave_times):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.sequence_length = sequence_length
        self.repeat_interleave_times = repeat_interleave_times

    def forward(self, x):
        x, _ = self.multihead_attn(x, x, x)
        x = torch.repeat_interleave(x, repeats=self.repeat_interleave_times, dim=1)
        return x


batch_size = 16
embed_dim = 256
num_heads = 8
sequence_length = 2048  # Increased from 512 to increase computation
repeat_interleave_times = 3


def get_inputs():
    x = torch.randn(batch_size, sequence_length, embed_dim)
    return [x]


def get_init_inputs():
    return [embed_dim, num_heads, sequence_length, repeat_interleave_times]