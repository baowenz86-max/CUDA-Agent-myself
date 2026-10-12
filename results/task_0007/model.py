import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, pos_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.pos_enc = nn.Parameter(torch.randn(1, pos_dim, embed_dim))

    def forward(self, query, key, value, attn_mask, target):
        q = self.W_q(query) + self.pos_enc[:, : query.shape[1], :]
        k = self.W_k(key) + self.pos_enc[:, : key.shape[1], :]
        v = self.W_v(value) + self.pos_enc[:, : value.shape[1], :]

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        logits = torch.baddbmm(torch.zeros_like(target), attn_output, attn_output.transpose(-2, -1))

        loss = self.bce_loss(logits, target)
        return loss


batch_size = 32
seq_len = 1024  # Increased sequence length to increase computation
embed_dim = 64
num_heads = 8
bias = True
pos_dim = 1024  # Matched pos_dim to new sequence length


def get_inputs():
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
    target = torch.randint(0, 2, (batch_size, seq_len, seq_len)).float()
    return [query, key, value, attn_mask, target]


def get_init_inputs():
    return [embed_dim, num_heads, bias, pos_dim]