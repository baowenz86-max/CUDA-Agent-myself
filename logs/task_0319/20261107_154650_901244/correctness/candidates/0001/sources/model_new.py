import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, dict_keys):
        super().__init__()
        # 匹配参考 state_dict 结构
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.param_dict = nn.ParameterDict(dict_keys)

    def forward(self, x):
        # 调用自定义算子 mha_forward，完全替代 torch 实现
        # 该算子将返回 tuple(attn_output, attn_weights)
        x, _ = torch.ops.cuda_extension.mha_forward(
            x,
            self.multihead_attention.in_proj_weight,
            self.multihead_attention.in_proj_bias,
            self.multihead_attention.out_proj.weight,
            self.multihead_attention.out_proj.bias,
        )
        return x

    def debug_forward(self, x):
        # 与 forward 相同的计算，但取回注意力权重
        attn_output, attn_weights = torch.ops.cuda_extension.mha_forward(
            x,
            self.multihead_attention.in_proj_weight,
            self.multihead_attention.in_proj_bias,
            self.multihead_attention.out_proj.weight,
            self.multihead_attention.out_proj.bias,
        )
        # 按检查点顺序产出
        yield "step_001", (attn_output, attn_weights)
        yield "step_002", attn_output
