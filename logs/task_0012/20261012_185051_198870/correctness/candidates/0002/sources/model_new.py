import torch
import torch.nn as nn

try:
    import cuda_extension
except ImportError:
    cuda_extension = None


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, input_shape):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.input_shape = input_shape

        # 保持与 nn.MultiheadAttention 相同的 state_dict 键和形状
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, bias=bias)

    def forward(self, x):
        # 步骤1: relu + sqrt
        x = cuda_extension.relu_sqrt_forward(x)
        # 步骤2: view 到 (seq_len, batch, embed_dim)
        x = x.view(self.input_shape[0], self.input_shape[1], self.embed_dim)
        # 步骤3: multihead attention
        x, _ = cuda_extension.mha_forward(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.num_heads
        )
        # 步骤4: view 到 (-1, embed_dim)
        x = x.view(-1, self.embed_dim)
        # 步骤5: silu
        x = cuda_extension.silu_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]

        # checkpoint 1: relu + sqrt
        x = cuda_extension.relu_sqrt_forward(x)
        yield "step_001", x

        # checkpoint 2: view
        x = x.view(self.input_shape[0], self.input_shape[1], self.embed_dim)
        yield "step_002", x

        # checkpoint 3: multihead attention 返回元组 (output, attn_weights)
        attn_output, attn_weights = cuda_extension.mha_forward(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.num_heads
        )
        x = attn_output
        yield "step_003", (x, attn_weights)

        # checkpoint 4: view
        x = x.view(-1, self.embed_dim)
        yield "step_004", x

        # checkpoint 5: silu
        x = cuda_extension.silu_forward(x)
        yield "step_005", x

        # checkpoint 6: return x
        yield "step_006", x
        return
