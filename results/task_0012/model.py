import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, input_shape):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, bias=bias)
        self.input_shape = input_shape

    def forward(self, x):
        x = torch.sqrt(torch.relu(x))  # Ensure non-negative input for sqrt
        x = x.view(self.input_shape[0], self.input_shape[1], self.embed_dim)  # Reshape for MultiheadAttention
        x, _ = self.multihead_attn(x, x, x)
        x = x.view(-1, self.embed_dim)  # Reshape back
        x = F.silu(x)
        return x


batch_size = 16
seq_length = 640  # Increased from 512 to 640 to increase computation time
embed_dim = 256
num_heads = 8
bias = True
input_shape = (seq_length, batch_size)  # Automatically updated with new seq_length


def get_inputs():
    x = torch.randn(batch_size, seq_length, embed_dim)
    return [x]


def get_init_inputs():
    return [embed_dim, num_heads, bias, input_shape]