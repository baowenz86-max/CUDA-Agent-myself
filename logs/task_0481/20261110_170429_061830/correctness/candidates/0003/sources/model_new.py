import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, linear_out_features, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear_out_features = linear_out_features

        # Build attention module with exact reference parameter names
        self.attention = nn.Module()
        self.attention.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.attention.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.attention.out_proj = nn.Module()
        self.attention.out_proj.weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.attention.out_proj.bias = nn.Parameter(torch.empty(embed_dim))

        # Linear module, matching reference
        self.linear = nn.Linear(embed_dim, linear_out_features, bias=bias)

        # Initialize parameters to match reference defaults
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.zeros_(self.attention.in_proj_bias)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        # nn.Linear already initializes with xavier_uniform and zeros

    def forward(self, query, key, value):
        # MultiheadAttention forward using custom op
        attn_result = cuda_extension.custom_multihead_attention(
            query, key, value,
            self.attention.in_proj_weight,
            self.attention.in_proj_bias,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
            self.embed_dim,
            self.num_heads,
            True,  # need_weights=True (reference default)
            True   # average_attn_weights=True (reference default)
        )
        attn_output, attn_weights = attn_result  # tuple unpacking

        # Concatenate attn_output with itself along last dim
        x = cuda_extension.custom_cat(attn_output, attn_output, -1)

        # GLU
        x = cuda_extension.custom_glu(x, -1)

        # Linear
        x = self.linear(x)

        return x

    def debug_forward(self, query, key, value):
        # Replicate forward but yield checkpoint values in order
        attn_result = cuda_extension.custom_multihead_attention(
            query, key, value,
            self.attention.in_proj_weight,
            self.attention.in_proj_bias,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
            self.embed_dim,
            self.num_heads,
            True,
            True
        )
        attn_output, attn_weights = attn_result

        # Yield step_001 value as tuple (attn_output, attn_weights)
        yield ('step_001', (attn_output, attn_weights))

        x = cuda_extension.custom_cat(attn_output, attn_output, -1)
        yield ('step_002', x)

        x = cuda_extension.custom_glu(x, -1)
        yield ('step_003', x)

        x = self.linear(x)
        yield ('step_004', x)

        yield ('step_005', x)
