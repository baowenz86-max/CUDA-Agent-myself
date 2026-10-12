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
        x = cuda_extension.mha_forward(
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
        attn_output = cuda_extension.mha_forward(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.num_heads
        )
        # 参考返回值是 (attn_output, attn_weights)，其中 attn_weights 需要从算子获取
        # 由于 mha_forward 可能只返回 output，我们需要设计算子返回两个值以匹配 checkpoint
        # 这里暂时假设 mha_forward 返回 (output, weights)
        x = attn_output
        attn_weights = cuda_extension.mha_forward_get_weights(
            x, x, x,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.num_heads
        )
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


# 为了正常 forward 也能使用相同的算子，不包含 debug 专用逻辑
# 自定义算子约定：
# relu_sqrt_forward(input) -> Tensor
# mha_forward(query, key, value, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, num_heads) -> Tensor
# silu_forward(input) -> Tensor
# 注意：mha_forward 仅返回输出张量；debug_forward 需要额外调用 mha_forward_get_weights 获取注意力权重。
