import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, repeat_interleave_times, scaling_factor):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.repeat_interleave_times = repeat_interleave_times
        self.scaling_factor = scaling_factor

    def forward(self, x, y):
        x = torch.repeat_interleave(x, repeats=self.repeat_interleave_times, dim=1)
        y = torch.repeat_interleave(y, repeats=self.repeat_interleave_times, dim=1)
        attn_output, attn_output_weights = self.multihead_attn(x, y, y)
        attn_output = torch.tan(attn_output * self.scaling_factor)
        return attn_output


batch_size = 128
embed_dim = 512
num_heads = 8
sequence_length = 50  # Increased from 10 to increase computation
repeat_interleave_times = 2
scaling_factor = 0.5


def get_inputs():
    x = torch.randn(batch_size, sequence_length, embed_dim)
    y = torch.randn(batch_size, sequence_length, embed_dim)
    return [x, y]


def get_init_inputs():
    return [embed_dim, num_heads, True, repeat_interleave_times, scaling_factor]