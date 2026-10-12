import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, input_shape):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.input_shape = input_shape

        # State dict keys must match nn.MultiheadAttention:
        # multihead_attn.in_proj_weight, multihead_attn.in_proj_bias,
        # multihead_attn.out_proj.weight, multihead_attn.out_proj.bias
        self.multihead_attn = nn.Module()
        self.multihead_attn.in_proj_weight = nn.Parameter(
            torch.empty(3 * embed_dim, embed_dim)
        )
        self.multihead_attn.in_proj_bias = nn.Parameter(
            torch.empty(3 * embed_dim)
        )
        self.multihead_attn.out_proj = nn.Module()
        self.multihead_attn.out_proj.weight = nn.Parameter(
            torch.empty(embed_dim, embed_dim)
        )
        self.multihead_attn.out_proj.bias = nn.Parameter(
            torch.empty(embed_dim)
        )

    def forward(self, x):
        x = cuda_extension.relu_sqrt_forward(x)
        x = x.view(self.input_shape[0], self.input_shape[1], self.embed_dim)
        x, _ = cuda_extension.mha_forward(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.num_heads,
        )
        x = x.view(-1, self.embed_dim)
        x = cuda_extension.silu_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu_sqrt_forward(x)
        yield "step_001", x
        x = x.view(self.input_shape[0], self.input_shape[1], self.embed_dim)
        yield "step_002", x
        mha_out, attn_weights = cuda_extension.mha_forward(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.num_heads,
        )
        yield "step_003", (mha_out, attn_weights)
        x = mha_out.view(-1, self.embed_dim)
        yield "step_004", x
        x = cuda_extension.silu_forward(x)
        yield "step_005", x
        yield "step_006", x
        return
