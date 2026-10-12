import torch
import torch.nn as nn
import math
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads, bias, pos_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        # 使用 nn.Linear 以保持 state_dict 键与参考一致
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.pos_enc = nn.Parameter(torch.randn(1, pos_dim, embed_dim))

    def forward(self, query, key, value, attn_mask, target):
        # 线性投影 q, k, v
        q = cuda_extension.linear_forward(query, self.W_q.weight, self.W_q.bias)
        k = cuda_extension.linear_forward(key, self.W_k.weight, self.W_k.bias)
        v = cuda_extension.linear_forward(value, self.W_v.weight, self.W_v.bias)

        # 添加位置编码
        q = cuda_extension.add_pos_enc(q, self.pos_enc)
        k = cuda_extension.add_pos_enc(k, self.pos_enc)
        v = cuda_extension.add_pos_enc(v, self.pos_enc)

        # 注意力
        attn_output = cuda_extension.scaled_dot_product_attention_forward(q, k, v, attn_mask)

        # logits = baddbmm(zeros_like(target), attn_output, attn_output^T)
        logits = cuda_extension.baddbmm_forward(torch.zeros_like(target), attn_output, attn_output.transpose(-2, -1))

        # 损失
        loss = cuda_extension.bce_with_logits_loss(logits, target)
        return loss

    def debug_forward(self, query, key, value, attn_mask, target):
        # 逐步计算并 yield 每个 checkpoint
        q = cuda_extension.linear_forward(query, self.W_q.weight, self.W_q.bias)
        q = cuda_extension.add_pos_enc(q, self.pos_enc)
        yield "step_001", q

        k = cuda_extension.linear_forward(key, self.W_k.weight, self.W_k.bias)
        k = cuda_extension.add_pos_enc(k, self.pos_enc)
        yield "step_002", k

        v = cuda_extension.linear_forward(value, self.W_v.weight, self.W_v.bias)
        v = cuda_extension.add_pos_enc(v, self.pos_enc)
        yield "step_003", v

        attn_output = cuda_extension.scaled_dot_product_attention_forward(q, k, v, attn_mask)
        yield "step_004", attn_output

        logits = cuda_extension.baddbmm_forward(torch.zeros_like(target), attn_output, attn_output.transpose(-2, -1))
        yield "step_005", logits

        loss = cuda_extension.bce_with_logits_loss(logits, target)
        yield "step_006", loss

        yield "step_007", loss
