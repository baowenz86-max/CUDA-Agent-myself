import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, linear_out_features, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Mimic reference MultiheadAttention structure
        self.attention = nn.Module()
        self.attention.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.attention.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.attention.out_proj = nn.Module()
        self.attention.out_proj.weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.attention.out_proj.bias = nn.Parameter(torch.empty(embed_dim))

        # Linear layer identical to reference
        self.linear = nn.Linear(embed_dim, linear_out_features, bias=bias)

        # Initialize parameters as reference does (xavier_uniform for weights, zeros for biases)
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.zeros_(self.attention.in_proj_bias)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, query, key, value):
        attn_result = cuda_extension.custom_multihead_attention(
            query, key, value,
            self.attention.in_proj_weight,
            self.attention.in_proj_bias,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
            self.embed_dim,
            self.num_heads,
            True,   # need_weights
            True    # average_attn_weights
        )
        attn_output, attn_weights = attn_result

        x = cuda_extension.custom_cat(attn_output, attn_output, -1)
        x = cuda_extension.custom_glu(x, -1)
        if self.linear.bias is not None:
            x = cuda_extension.custom_linear(x, self.linear.weight, self.linear.bias)
        else:
            x = cuda_extension.custom_linear(x, self.linear.weight, None)
        return x

    def debug_forward(self, query, key, value):
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
        yield ('step_001', (attn_output, attn_weights))

        x = cuda_extension.custom_cat(attn_output, attn_output, -1)
        yield ('step_002', x)

        x = cuda_extension.custom_glu(x, -1)
        yield ('step_003', x)

        if self.linear.bias is not None:
            x = cuda_extension.custom_linear(x, self.linear.weight, self.linear.bias)
        else:
            x = cuda_extension.custom_linear(x, self.linear.weight, None)
        yield ('step_004', x)
        yield ('step_005', x)
