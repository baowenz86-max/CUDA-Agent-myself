import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, batch_first, stack_dim):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=batch_first)
        self.stack_dim = stack_dim

    def forward(self, x1, x2, x3, key_padding_mask):
        # Use custom multihead attention op that fully replaces nn.MultiheadAttention forward
        attn_output = cuda_extension.multihead_attention_forward(
            x1, x2, x3,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            key_padding_mask,
            self.multihead_attn.embed_dim,
            self.multihead_attn.num_heads,
            self.multihead_attn.batch_first
        )
        # Elementwise multiply with x2 (custom op)
        x4 = cuda_extension.multiply(attn_output, x2)
        # Stack along specified dimension (custom op)
        return cuda_extension.stack([x3, x4], dim=self.stack_dim)
