import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, loss_target_shape):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.loss_target = nn.Parameter(torch.randn(loss_target_shape))

    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        cat_output = torch.cat([attn_output, query], dim=2)
        loss = F.smooth_l1_loss(cat_output, self.loss_target)
        return cat_output, attn_output_weights, loss


batch_size = 16
sequence_length = 1024  # Increased from 512 to increase computation
embed_dim = 256
num_heads = 8


def get_inputs():
    query = torch.randn(batch_size, sequence_length, embed_dim)
    key = torch.randn(batch_size, sequence_length, embed_dim)
    value = torch.randn(batch_size, sequence_length, embed_dim)
    return [query, key, value]


def get_init_inputs():
    loss_target_shape = (batch_size, sequence_length, embed_dim * 2)
    return [embed_dim, num_heads, True, loss_target_shape]