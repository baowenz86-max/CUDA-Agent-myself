import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, pos_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.pos_dim = pos_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.pos_enc = nn.Parameter(torch.randn(1, pos_dim, embed_dim))

    def forward(self, query, key, value, attn_mask, target):
        # Ensure inputs are contiguous and on CUDA (custom ops assume this)
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        attn_mask = attn_mask.contiguous()
        target = target.contiguous()

        # Custom ops
        qkv = cuda_extension.qkv_proj_pos(
            query, key, value, self.pos_enc,
            self.W_q.weight, self.W_k.weight, self.W_v.weight,
            self.W_q.bias if self.bias else None,
            self.W_k.bias if self.bias else None,
            self.W_v.bias if self.bias else None
        )
        q, k, v = qkv  # each [B, T, E]

        attn_output = cuda_extension.sdpa(q, k, v, attn_mask)

        logits = cuda_extension.attn_bias_baddbmm(attn_output)

        loss = cuda_extension.bce_with_logits(logits, target)
        return loss
