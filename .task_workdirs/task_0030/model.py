import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(self, magnitudes, angles, query, key, value):
        complex_numbers = torch.polar(magnitudes, angles)
        one_hot_encodings = F.one_hot(torch.arange(self.num_classes), num_classes=self.num_classes).float()
        attention_output = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        return complex_numbers, one_hot_encodings, attention_output


batch_size = 512
num_classes = 10
num_heads = 8
embed_dim = 128  # Increased from 64 to 128
seq_length = 128  # Increased from 32 to 128

magnitudes = torch.randn(batch_size, seq_length, embed_dim)
angles = torch.randn(batch_size, seq_length, embed_dim)
query = torch.randn(batch_size, num_heads, seq_length, embed_dim // num_heads)
key = torch.randn(batch_size, num_heads, seq_length, embed_dim // num_heads)
value = torch.randn(batch_size, num_heads, seq_length, embed_dim // num_heads)


def get_inputs():
    return [magnitudes, angles, query, key, value]


def get_init_inputs():
    return [num_classes, embed_dim, num_heads]