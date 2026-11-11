import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x, attn_mask=None):
        attn_output = cuda_extension.multihead_attention_forward(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            attn_mask,
            self.multihead_attn.embed_dim,
            self.multihead_attn.num_heads
        )
        return cuda_extension.mean_forward(attn_output, 1)

    def debug_forward(self, x, attn_mask=None):
        attn_output = cuda_extension.multihead_attention_forward(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            attn_mask,
            self.multihead_attn.embed_dim,
            self.multihead_attn.num_heads
        )
        final_output = cuda_extension.mean_forward(attn_output, 1)
        yield "step_001", final_output
