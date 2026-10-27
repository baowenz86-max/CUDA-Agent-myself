import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, pad, embed_dim, num_heads, bias):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias

    def forward(self, q, k, v):
        q = self.reflection_pad(q)
        k = self.reflection_pad(k)
        v = self.reflection_pad(v)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=self.bias)
        return attn_output


pad = (1, 1, 1, 1)
embed_dim = 256
num_heads = 8
bias = False
batch_size = 32
seq_length = 1024  # Increased from 512 to increase runtime


def get_inputs():
    head_dim = embed_dim // num_heads
    q = torch.randn(batch_size, num_heads, seq_length, head_dim)
    k = torch.randn(batch_size, num_heads, seq_length, head_dim)
    v = torch.randn(batch_size, num_heads, seq_length, head_dim)
    return [q, k, v]


def get_init_inputs():
    return [pad, embed_dim, num_heads, bias]