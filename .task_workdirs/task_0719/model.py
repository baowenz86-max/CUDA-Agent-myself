import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = F.softmax
        self.scaled_dot_product_attention = F.scaled_dot_product_attention

    def forward(self, query, key, value):
        query = self.softmax(query, dim=-1)
        output = self.scaled_dot_product_attention(query, key, value)
        return output


batch_size = 512
seq_len = 384  # Increased from 128 to increase computational load
embed_dim = 64


def get_inputs():
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    return [query, key, value]


def get_init_inputs():
    return []