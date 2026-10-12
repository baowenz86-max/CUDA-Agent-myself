import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, repeat_interleave_times, scaling_factor):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.repeat_interleave_times = repeat_interleave_times
        self.scaling_factor = scaling_factor
        self.num_heads = num_heads

    def forward(self, x, y):
        out = cuda_extension.fused_attention_forward(
            x, y,
            self.multihead_attn.in_proj_weight, self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight, self.multihead_attn.out_proj.bias,
            self.repeat_interleave_times, self.num_heads, self.scaling_factor
        )
        return out
