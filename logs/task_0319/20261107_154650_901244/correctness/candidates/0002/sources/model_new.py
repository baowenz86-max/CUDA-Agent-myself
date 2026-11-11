import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, dict_keys):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.param_dict = nn.ParameterDict(dict_keys)

    def forward(self, x):
        x, _ = cuda_extension.mha_forward(
            x,
            self.multihead_attention.in_proj_weight,
            self.multihead_attention.in_proj_bias,
            self.multihead_attention.out_proj.weight,
            self.multihead_attention.out_proj.bias,
        )
        return x

    def debug_forward(self, x):
        attn_output, attn_weights = cuda_extension.mha_forward(
            x,
            self.multihead_attention.in_proj_weight,
            self.multihead_attention.in_proj_bias,
            self.multihead_attention.out_proj.weight,
            self.multihead_attention.out_proj.bias,
        )
        yield "step_001", (attn_output, attn_weights)
        yield "step_002", attn_output
