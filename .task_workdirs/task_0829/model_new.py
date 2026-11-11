import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        # 保留参数名称与参考 state_dict 完全一致
        # 注意：nn.MultiheadAttention 内部会创建 in_proj_weight, in_proj_bias, out_proj.weight, out_proj.bias
        # 我们直接使用该模块的参数，不复制，以保持 state_dict 键和形状。

    def forward(self, query, key, value):
        # 调用自定义扩展算子执行多头注意力前向
        # 算子返回 (output, attn_weights) 与 nn.MultiheadAttention 前向一致
        x, attn_weights = cuda_extension.multihead_attention_forward(
            query, key, value,
            self.attention.in_proj_weight, self.attention.in_proj_bias,
            self.attention.out_proj.weight, self.attention.out_proj.bias,
            self.attention.embed_dim, self.attention.num_heads,
            self.attention.batch_first,
            False,  # add_zero_attn
            False,  # is_causal
            None,   # key_padding_mask
            None,   # attn_mask
            True,   # need_weights
            True    # average_attn_weights
        )
        # 调用自定义矩阵乘法算子
        out = cuda_extension.matmul_forward(x, value.transpose(1, 2))
        return out

    def debug_forward(self, *inputs):
        query, key, value = inputs
        x, attn_weights = cuda_extension.multihead_attention_forward(
            query, key, value,
            self.attention.in_proj_weight, self.attention.in_proj_bias,
            self.attention.out_proj.weight, self.attention.out_proj.bias,
            self.attention.embed_dim, self.attention.num_heads,
            self.attention.batch_first,
            False,  # add_zero_attn
            False,  # is_causal
            None,   # key_padding_mask
            None,   # attn_mask
            True,   # need_weights
            True    # average_attn_weights
        )
        yield "step_001", (x, attn_weights)
        out = cuda_extension.matmul_forward(x, value.transpose(1, 2))
        yield "step_002", out
