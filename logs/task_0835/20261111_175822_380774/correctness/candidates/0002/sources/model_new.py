import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first):
        super().__init__()
        # 保存参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        # 创建与 nn.MultiheadAttention 相同结构的参数容器，保留 state_dict 键
        self.multihead_attn = nn.Module()
        qkv_dim = 3 * embed_dim
        self.multihead_attn.in_proj_weight = nn.Parameter(torch.empty(qkv_dim, embed_dim))
        self.multihead_attn.in_proj_bias = nn.Parameter(torch.empty(qkv_dim))
        self.multihead_attn.out_proj = nn.Module()
        self.multihead_attn.out_proj.weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.multihead_attn.out_proj.bias = nn.Parameter(torch.empty(embed_dim))
        # 初始化与 nn.MultiheadAttention 内部一致
        nn.init.xavier_uniform_(self.multihead_attn.in_proj_weight)
        nn.init.constant_(self.multihead_attn.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.multihead_attn.out_proj.weight)
        nn.init.constant_(self.multihead_attn.out_proj.bias, 0.0)

    def forward(self, query, key, value):
        # 调用自定义 multihead attention 算子
        attn_output, _ = cuda_extension.multihead_attn_forward(
            query, key, value,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.embed_dim,
            self.num_heads,
            self.batch_first
        )
        # 调用自定义 log10_abs_eps 算子
        output = cuda_extension.log10_abs_eps(attn_output, 1e-6)
        return output

    def debug_forward(self, query, key, value):
        # 按照 checkpoints 顺序 yield (id, value)
        attn_output, attn_output_weights = cuda_extension.multihead_attn_forward(
            query, key, value,
            self.multihead_attn.in_proj_weight,
            self.multihead_attn.in_proj_bias,
            self.multihead_attn.out_proj.weight,
            self.multihead_attn.out_proj.bias,
            self.embed_dim,
            self.num_heads,
            self.batch_first
        )
        yield "step_001", (attn_output, attn_output_weights)
        output = cuda_extension.log10_abs_eps(attn_output, 1e-6)
        yield "step_002", output
