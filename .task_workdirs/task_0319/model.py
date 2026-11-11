import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, dict_keys):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.param_dict = nn.ParameterDict(dict_keys)

    def forward(self, x):
        x, _ = self.multihead_attention(x, x, x)
        return x


embed_dim = 512
num_heads = 4
dict_keys = {"param1": nn.Parameter(torch.randn(embed_dim))}


def get_inputs():
    # Increased sequence length from 32 to 256 to increase computation time
    return [torch.randn(256, 32, embed_dim)]


def get_init_inputs():
    return [embed_dim, num_heads, dict_keys]